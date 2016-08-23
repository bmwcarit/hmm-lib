/**
 * Copyright (C) 2016, BMW AG
 * Author: Stefan Holder (stefan.holder@bmw.de)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.bmw.hmm;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

/**
 * Computes the forward-backward algorithm, also known as smoothing.
 * This algorithm computes the probability of each state candidate at each time step given the
 * entire observation sequence.
 *
 * @param <S> the state type
 * @param <O> the observation type
 */
public class ForwardBackwardAlgorithm<S, O> {

    /**
     * Internal state of each time step.
     */
    private class Step {
        final Collection<S> candidates;
        final Map<S, Double> emissionProbabilities;
        final Map<Transition<S>, Double> transitionProbabilities;
        final Map<S, Double> forwardProbabilities;
        final double scalingDivisor; // Normalizes sum of forward probabilities to 1.

        Step(Collection<S> candidates, Map<S, Double> emissionProbabilities,
                Map<Transition<S>, Double> transitionProbabilities,
                Map<S, Double> forwardProbabilities, double scalingDivisor) {
            this.candidates = candidates;
            this.emissionProbabilities = emissionProbabilities;
            this.transitionProbabilities = transitionProbabilities;
            this.forwardProbabilities = forwardProbabilities;
            this.scalingDivisor = scalingDivisor;
        }
    }

    private static final double DELTA = 1e-8;

    private List<Step> steps;
    private Collection<S> prevCandidates; // For on-the-fly computation of forward probabilities

    /**
     * Lets the computation start with the given initial state probabilities.
     *
     * @param initialStates Pass a collection with predictable iteration order such as
     * {@link ArrayList} to ensure deterministic results.
     *
     * @param initialProbabilities Initial probabilities for each initial state.
     *
     * @throws NullPointerException if any initial probability is missing
     *
     * @throws IllegalStateException if this method or
     * {@link #startWithInitialObservation(Object, Collection, Map)} has already been called
     */
    public void startWithInitialStateProbabilities(Collection<S> initialStates,
            Map<S, Double> initialProbabilities) {
        if (!sumsToOne(initialProbabilities.values())) {
            throw new IllegalArgumentException("Initial state probabilities must sum to 1.");
        }

        initializeStateProbabilities(null, initialStates, initialProbabilities);
    }

    /**
     * Lets the computation start at the given first observation.
     *
     * @param candidates Pass a collection with predictable iteration order such as
     * {@link ArrayList} to ensure deterministic results.
     *
     * @param emissionProbabilities Emission probabilities of the first observation for
     * each of the road position candidates.
     *
     * @throws NullPointerException if any emission probability is missing
     *
     * @throws IllegalStateException if this method or
     * {@link #startWithInitialStateProbabilities(Collection, Map)}} has already been called
     */
    public void startWithInitialObservation(O observation, Collection<S> candidates,
            Map<S, Double> emissionProbabilities) {
        initializeStateProbabilities(observation, candidates, emissionProbabilities);
    }

    /**
     * Processes the next time step.
     *
     * @param candidates Pass a collection with predictable iteration order such as
     * {@link ArrayList} to ensure deterministic results.
     *
     * @param emissionProbabilities Emission probabilities for each candidate state.
     *
     * @param transitionProbabilities Transition probability between all pairs of candidates.
     * A transition probability of zero is assumed for every missing transition.
     *
     * @throws NullPointerException if any emission probability is missing
     *
     * @throws IllegalStateException if neither
     * {@link #startWithInitialStateProbabilities(Collection, Map)} nor
     * {@link #startWithInitialObservation(Object, Collection, Map)} has not been called before
     */
    public void nextStep(O observation, Collection<S> candidates,
            Map<S, Double> emissionProbabilities,
            Map<Transition<S>, Double> transitionProbabilities) {
        if (steps == null) {
            throw new IllegalStateException("startWithInitialStateProbabilities(...) or " +
                    "startWithInitialObservation(...) must be called first.");
        }

        // Make defensive copies.
        candidates = new ArrayList<>(candidates);
        emissionProbabilities = new LinkedHashMap<>(emissionProbabilities);
        transitionProbabilities = new LinkedHashMap<>(transitionProbabilities);

        // On-the-fly computation of forward probabilities at each step allows to efficiently
        // (re)compute smoothing probabilities at any time step.
        final Map<S, Double> prevForwardProbabilities =
                steps.get(steps.size() - 1).forwardProbabilities;
        final Map<S, Double> curForwardProbabilities = new LinkedHashMap<>();
        double sum = 0.0;
        for (S curState : candidates) {
            final double forwardProbability = computeForwardProbability(curState,
                    prevForwardProbabilities, emissionProbabilities, transitionProbabilities);
            curForwardProbabilities.put(curState, forwardProbability);
            sum += forwardProbability;
        }

        normalizeForwardProbabilities(curForwardProbabilities, sum);
        steps.add(new Step(candidates, emissionProbabilities, transitionProbabilities,
                curForwardProbabilities, sum));

        prevCandidates = candidates;
    }

    /**
     * Returns the probability for all candidates of all time steps given all observations.
     * The time steps include the initial states/observations time step.
     */
    public List<Map<S, Double>> computeSmoothingProbabilities() {
        return computeSmoothingProbabilities(null);
    }

    /**
     * Returns the probability of the specified candidate at the specified zero-based time step
     * given the observations up to t.
     */
    public double forwardProbability(int t, S candidate) {
        if (steps == null) {
            throw new IllegalStateException("No time steps yet.");
        }

        return steps.get(t).forwardProbabilities.get(candidate);
    }

    /**
     * Returns the probability of the specified candidate given all previous observations.
     */
    public double currentForwardProbability(S candidate) {
        if (steps == null) {
            throw new IllegalStateException("No time steps yet.");
        }

        return forwardProbability(steps.size() - 1, candidate);
    }

    /**
     * Returns the log probability of the entire observation sequence.
     * The log is returned to prevent arithmetic underflows for very small probabilities.
     */
    public double observationLogProbability() {
        if (steps == null) {
            throw new IllegalStateException("No time steps yet.");
        }

        double result = 0.0;
        for (Step step : steps) {
            result += Math.log(step.scalingDivisor);
        }
        return result;
    }

    /**
     * @see #computeSmoothingProbabilities()
     *
     * @param outBackwardProbabilities optional output parameter for backward probabilities,
     * must be empty if not null.
     */
    List<Map<S, Double>> computeSmoothingProbabilities(
            List<Map<S, Double>> outBackwardProbabilities) {
        assert outBackwardProbabilities == null || outBackwardProbabilities.isEmpty();

        final List<Map<S, Double>> result = new ArrayList<>();

        ListIterator<Step> stepIter = steps.listIterator(steps.size());
        if (!stepIter.hasPrevious()) {
            return result;
        }

        // Initial step
        Step step = stepIter.previous();
        Map<S, Double> backwardProbabilities = new LinkedHashMap<>();
        for (S candidate : step.candidates) {
            backwardProbabilities.put(candidate, 1.0);
        }
        if (outBackwardProbabilities != null) {
            outBackwardProbabilities.add(backwardProbabilities);
        }
        result.add(computeSmoothingProbabilitiesVector(step.candidates, step.forwardProbabilities,
                backwardProbabilities));

        // Remaining steps
        while (stepIter.hasPrevious()) {
            final Step nextStep = step;
            step = stepIter.previous();
            final Map<S, Double> nextBackwardProbabilities = backwardProbabilities;
            backwardProbabilities = new LinkedHashMap<>();
            for (S candidate : step.candidates) {
                // Using the scaling divisors of the next steps eliminates the need to
                // normalize the smoothing probabilities,
                // see also https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm.
                final double probability = computeUnscaledBackwardProbability(candidate,
                        nextBackwardProbabilities, nextStep) / nextStep.scalingDivisor;
                backwardProbabilities.put(candidate, probability);
            }
            if (outBackwardProbabilities != null) {
                outBackwardProbabilities.add(backwardProbabilities);
            }
            result.add(computeSmoothingProbabilitiesVector(step.candidates,
                    step.forwardProbabilities, backwardProbabilities));
        }
        Collections.reverse(result);
        return result;
    }

    private Map<S, Double> computeSmoothingProbabilitiesVector(Collection<S> candidates,
            Map<S, Double> forwardProbabilities, Map<S, Double> backwardProbabilities) {
        assert forwardProbabilities.size() == backwardProbabilities.size();
        final Map<S, Double> result = new LinkedHashMap<>();
        for (S state : candidates) {
            final double probability = forwardProbabilities.get(state)
                    * backwardProbabilities.get(state);
            assert Utils.probabilityInRange(probability, DELTA);
            result.put(state, probability);
        }
        assert sumsToOne(result.values());
        return result;
    }

    private double computeUnscaledBackwardProbability(S candidate,
            Map<S, Double> nextBackwardProbabilities, Step nextStep) {
        double result = 0.0;
        for (S nextCandidate : nextStep.candidates) {
            result += nextStep.emissionProbabilities.get(nextCandidate) *
                    nextBackwardProbabilities.get(nextCandidate) * transitionProbability(
                    candidate, nextCandidate, nextStep.transitionProbabilities);
        }
        return result;
    }

    private boolean sumsToOne(Collection<Double> probabilities) {
        double sum = 0.0;
        for (double probability : probabilities) {
            sum += probability;
        }
        return Math.abs(sum - 1.0) <= DELTA;
    }

    /**
     * @param observation Use only if HMM only starts with first observation.
     */
    private void initializeStateProbabilities(O observation, Collection<S> candidates,
            Map<S, Double> initialProbabilities) {
        if (steps != null) {
            throw new IllegalStateException("Initial probabilities have already been set.");
        }

        candidates = new ArrayList<>(candidates); // Defensive copy
        steps = new ArrayList<>();

        final Map<S, Double> forwardProbabilities = new LinkedHashMap<>();
        double sum = 0.0;
        for (S candidate : candidates) {
            final double forwardProbability = initialProbabilities.get(candidate);
            forwardProbabilities.put(candidate, forwardProbability);
            sum += forwardProbability;
        }

        normalizeForwardProbabilities(forwardProbabilities, sum);
        steps.add(new Step(candidates, null, null, forwardProbabilities, sum));

        prevCandidates = candidates;
    }

    /**
     * Returns the non-normalized forward probability of the specified state.
     */
    private double computeForwardProbability(S curState,
            Map<S, Double> prevForwardProbabilities, Map<S, Double> emissionProbabilities,
            Map<Transition<S>, Double> transitionProbabilities) {
        double result = 0.0;
        for (S prevState : prevCandidates) {
            result += prevForwardProbabilities.get(prevState) *
                    transitionProbability(prevState, curState, transitionProbabilities);
        }
        result *= emissionProbabilities.get(curState);
        return result;
    }

    /**
     * Returns zero probability for non-existing transitions.
     */
    private double transitionProbability(S prevState, S curState,
            Map<Transition<S>, Double> transitionProbabilities) {
        final Double transitionProbability =
                transitionProbabilities.get(new Transition<S>(prevState, curState));
        return transitionProbability == null ? 0.0 : transitionProbability;
    }

    private void normalizeForwardProbabilities(
            Map<S, Double> forwardProbabilities, double sum) {
        for (Map.Entry<S, Double> entry : forwardProbabilities.entrySet()) {
            forwardProbabilities.put(entry.getKey(), entry.getValue() / sum);
        }
    }

}
