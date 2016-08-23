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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Objects;

/**
 * Computes the forward / backward algorithm, also known as smoothing.
 * This algorithm computes the probability of each state candidate given the entire observation
 * sequence.
 *
 * @param <S> the state type
 * @param <O> the observation type
 */
public class ForwardBackwardAlgorithm<S, O> {

    private class Step {
        final Collection<S> candidates;
        final Map<S, Double> emissionProbabilities;
        final Map<Transition<S>, Double> transitionProbabilities;
        final Map<S, Double> forwardProbabilities;
        final double scalingDivisor;
        Map<S, Double> posteriorProbabilities = null;

        @SuppressWarnings("unused")
        Map<S, Double> backwardProbabilities = null;

        Step(Collection<S> candidates, Map<S, Double> emissionProbabilities,
                Map<Transition<S>, Double> transitionProbabilities,
                Map<S, Double> forwardProbabilities, double scalingDivisor) {
            this.candidates = candidates;
            this.emissionProbabilities = emissionProbabilities;
            this.transitionProbabilities = transitionProbabilities;
            this.forwardProbabilities = forwardProbabilities;
            this.scalingDivisor = scalingDivisor;
        }

        public void setBackwardProbabilities(Map<S, Double> backwardProbabilities) {
            assert forwardProbabilities.size() == backwardProbabilities.size();
            this.backwardProbabilities = backwardProbabilities;
            posteriorProbabilities = new LinkedHashMap<>();
            for (S state : candidates) {
                final double probability = forwardProbabilities.get(state)
                        * backwardProbabilities.get(state);
                posteriorProbabilities.put(state, probability);
            }
            assert sumsToOne(posteriorProbabilities.values());
        }

    }

    private List<Step> steps;
    private Collection<S> prevCandidates; // for on-the-fly computation of forward probabilities
    private boolean computedPosteriorProbabilities = false;

    public void startWithInitialStateProbabilities(Collection<S> initialStates,
    		Map<S, Double> initialProbabilities) {
        if (!sumsToOne(initialProbabilities.values())) {
            throw new IllegalArgumentException("Initial state probabilities must sum to 1.");
        }

    	initializeStateProbabilities(null, initialStates, initialProbabilities);
    }

    public void startWithInitialObservation(O observation, Collection<S> candidates,
    		Map<S, Double> emissionProbabilities) {
    	initializeStateProbabilities(observation, candidates, emissionProbabilities);
    }

    public void nextStep(O observation, Collection<S> candidates,
    		Map<S, Double> emissionProbabilities,
    		Map<Transition<S>, Double> transitionProbabilities) {
    	Objects.requireNonNull(steps,
    			"startWithInitialStateProbabilities() or startWithInitialObservation() "
    				+ "must be called first.");

    	// Note that on-the-fly computation of forward probabilities enables on-the-fly
    	// checking and handling of HMM breaks.
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
    }

    public void computePosteriorProbabilities() {
        if (computedPosteriorProbabilities) {
            throw new IllegalStateException("Backward probabilities have already been computed.");
        }

        ListIterator<Step> stepIter = steps.listIterator(steps.size());

        if (!stepIter.hasPrevious()) return;

        // Initial step
        Step step = stepIter.previous();
        Map<S, Double> backwardProbabilities = new LinkedHashMap<>();
        for (S candidate : step.candidates) {
            backwardProbabilities.put(candidate, 1.0);
        }
        step.setBackwardProbabilities(backwardProbabilities);

        // Remaining steps
        while (stepIter.hasPrevious()) {
            Step nextStep = step;
            step = stepIter.previous();
            Map<S, Double> nextBackwardProbabilities = backwardProbabilities;
            backwardProbabilities = new LinkedHashMap<>();
            for (S candidate : step.candidates) {
                final double probability = computeUnscaledBackwardProbability(candidate,
                        nextBackwardProbabilities, nextStep) / nextStep.scalingDivisor;
                backwardProbabilities.put(candidate, probability);
            }
            step.setBackwardProbabilities(backwardProbabilities);
        }

        computedPosteriorProbabilities = true;
    }

    private double computeUnscaledBackwardProbability(S candidate,
            Map<S, Double> nextBackwardProbabilities, Step nextStep) {
        double result = 0.0;
        for (S nextCandidate : nextStep.candidates) {
            result += nextStep.emissionProbabilities.get(nextCandidate) *
                    nextBackwardProbabilities.get(nextCandidate) *
                    nextStep.transitionProbabilities.get(
                            new Transition<S>(candidate, nextCandidate));
        }
        // Divide by scaling divisor of nextStep
        return result;
    }

    /**
     * Returns the probability of the specified candidate at the specified zero-based time step
     * given the observations up to t.
     */
    public double forwardProbability(int t, S candidate) {
        Objects.requireNonNull(steps, "No time steps yet.");

    	return steps.get(t).forwardProbabilities.get(candidate);
    }

    /**
     * Returns the probability of the specified candidate given all previous observations.
     */
    public double currentForwardProbability(S candidate) {
        Objects.requireNonNull(steps, "No time steps yet.");

        return forwardProbability(steps.size() - 1, candidate);
    }

    /**
     * Returns the probability of the specified candidate at the specified zero-based time step
     * given all observations.
     */
    public double posteriorProbability(int t, S candidate) {
        if (!computedPosteriorProbabilities) {
            throw new IllegalStateException("Posterior probabilties must be computed first.");
        }

        return steps.get(t).posteriorProbabilities.get(candidate);
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

    private boolean sumsToOne(Collection<Double> probabilities) {
        final double DELTA = 1e-8;
        double sum = 0.0;
        for (double probability : probabilities) {
            sum += probability;
        }
        return Math.abs(sum - 1.0) < DELTA;
    }

    /**
     * @param observation Use only if HMM only starts with first observation.
     */
    private void initializeStateProbabilities(O observation, Collection<S> candidates,
    		Map<S, Double> initialProbabilities) {
    	if (steps != null) {
    		throw new IllegalStateException("Initial probabilities have already been set.");
    	}

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
					transitionProbabilities.get(new Transition<S>(prevState, curState));
		}
		result *= emissionProbabilities.get(curState);
		return result;
	}

	private void normalizeForwardProbabilities(
			Map<S, Double> forwardProbabilities, double sum) {
		for (Map.Entry<S, Double> entry : forwardProbabilities.entrySet()) {
    		forwardProbabilities.put(entry.getKey(), entry.getValue() / sum);
    	}
	}

}
