/**
 * Copyright (C) 2015-2016, BMW Car IT GmbH and BMW AG
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

package com.bmw.hmm_lib;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Implementation of the Viterbi algorithm for time-inhomogeneous Markov processes, 
 * meaning that the set of states and state transition probabilities are not necessarily fixed
 * for all time steps. The plain Viterbi algorithm for stationary Markov processes is described e.g.
 * in Rabiner, Juang, An introduction to Hidden Markov Models, IEEE ASSP Mag., pp 4-16, June 1986.
 * 
 * <p>Generally expects logarithmic probabilities as input to prevent arithmetic underflows for
 * small probability values. 
 * 
 * <p>This algorithm supports storing transition objects in
 * {@link ViterbiAlgorithm#nextStep(Object, Collection, Map, Map, Map)}. For instance if a HMM is
 * used for map matching, this could be routes between road position candidates.
 * The transition descriptors of the most likely sequence can be retrieved later in 
 * {@link SequenceState#transitionDescriptor} and hence do not need to be stored by the
 * caller. Since the caller does not know in advance which transitions will occur in the most
 * likely sequence, this reduces the number of transitions that need to be kept in memory
 * from t*n² to t*n since only one transition descriptor is stored per back pointer, 
 * where t is the number of time steps and n the number of candidates per time step.
 * 
 * <p>For long observation sequences, back pointers usually converge to a single path after a
 * certain number of time steps. For instance, when matching GPS coordinates to roads, the last
 * GPS positions in the trace usually do not affect the first road matches anymore.
 * This implementation exploits this fact by letting the Java garbage collector
 * take care of unreachable back pointers. If back pointers converge to a single path after a
 * constant number of time steps, only O(t) back pointers and transition descriptors need to be
 * stored in memory.
 * 
 * @param <S> the state type
 * @param <O> the observation type
 * @param <D> the transition descriptor type. Pass {@link Object} if transition descriptors are not
 * needed.
 */
public class ViterbiAlgorithm<S, O, D> {

    /**
     * Stores addition information for each candidate.
     */
    private static class ExtendedState<S, O, D> {
        
        S state;

        /**
         * Back pointer to previous state candidate in the most likely sequence.
         * See {@link MostLikelySequence#backPointerSequence}.
         * Back pointers are chained using plain Java references.
         * This allows garbage collection of unreachable back pointers.
         */
        ExtendedState<S, O, D> backPointer;
        
        O observation;
        D transitionDescriptor;
        
        ExtendedState(S state,
                ExtendedState<S, O, D> backPointer,
                O observation, D transitionDescriptor) {
            this.state = state;
            this.backPointer = backPointer;
            this.observation = observation;
            this.transitionDescriptor = transitionDescriptor;
        }
    }
    
    private static class ForwardStepResult<S, O, D> {
        /**
         * Log probability of each state. See {@link MostLikelySequence#messageHistory}.
         */
        final Map<S, Double> message;

        /**
         * Back pointers to previous state candidates for retrieving the most likely sequence after
         * the forward pass. See {@link MostLikelySequence#backPointerSequence}
         */
        final Map<S, ExtendedState<S, O, D>> backPointers;

        ForwardStepResult(int numberStates) {
            message = new LinkedHashMap<>(Utils.initialHashMapCapacity(numberStates));
            backPointers = new LinkedHashMap<>(Utils.initialHashMapCapacity(numberStates));
        }
    }
    
    /**
     * Allows to retrieve the most likely sequence using back pointers.
     */
    private Map<S, ExtendedState<S, O, D>> lastExtendedStates;
    
    private Collection<S> prevCandidates;
    private Map<S, Double> message;
    private boolean isBroken = false;
    
    private List<Map<S, Double>> messageHistory;
    
    /**
     * Need to construct a new instance for each sequence of observations.
     * Does not keep the message history.
     */
    public ViterbiAlgorithm() {
        this(false);
    }
    
    /**
     * Need to construct a new instance for each sequence of observations.
     * @param keepMessageHistory Whether to store intermediate forward messages
     * (probabilities of intermediate most likely paths) for debugging.
     */
    public ViterbiAlgorithm(boolean keepMessageHistory) {
        if (keepMessageHistory) {
            messageHistory = new ArrayList<Map<S, Double>>();
        }
    }

    /**
     * Lets the HMM computation start with the given initial state probabilities. 
     * 
     * @param initialStates Pass a collection with predictable iteration order such as
     * {@link ArrayList} to ensure deterministic results.
     * @param initialLogProbabilities Initial log probabilities for each initial state.
     * 
     * @throws NullPointerException if any initial probability is missing
     * 
     * @throws IllegalStateException if this method or 
     * {@link ViterbiAlgorithm#startWithInitialObservation(Object, Collection, Map)} 
     * has already been called  
     */
    public void startWithInitialStateProbabilities(Collection<S> initialStates, 
            Map<S, Double> initialLogProbabilities) {
        initializeStateProbabilities(null, initialStates, initialLogProbabilities);
    }
    
    /**
     * Lets the HMM computation starts at the given first observation and uses the given emission
     * probabilities as the initial state probability for each starting state s. 
     * 
     * @param candidates Pass a collection with predictable iteration order such as
     * {@link ArrayList} to ensure deterministic results.
     * @param emissionLogProbabilities Emission log probabilities of the first observation for
     * each of the road position candidates.
     * 
     * @throws NullPointerException if any emission probability is missing
     * 
     * @throws IllegalStateException if this method or 
     * {@link ViterbiAlgorithm#startWithInitialStateProbabilities(Collection, Map)}} 
     * has already been called  
     */
    public void startWithInitialObservation(O observation, Collection<S> candidates, 
            Map<S, Double> emissionLogProbabilities) {
        initializeStateProbabilities(observation, candidates, emissionLogProbabilities);
    }
    
    /**
     * Processes the next time step. Must not be called if the HMM is broken. 
     * 
     * @param candidates Pass a collection with predictable iteration order such as
     * {@link ArrayList} to ensure deterministic results.
     * @param emissionLogProbabilities Emission log probabilities for each candidate state.
     *  
     * @param transitionLogProbabilities Transition log probability between all pairs of candidates.
     * A transition probability of zero is assumed for every missing transition. 
     * 
     * @param transitionDescriptors Optional objects that describes the transitions.
     * 
     * @throws NullPointerException if any emission probability is missing
     * 
     * @throws IllegalStateException if neither 
     * {@link ViterbiAlgorithm#startWithInitialStateProbabilities(Collection, Map)} nor
     * {@link ViterbiAlgorithm#startWithInitialObservation(Object, Collection, Map)}
     * has not been called before or if this method is called after an HMM break has occurred 
     */
    public void nextStep(O observation, Collection<S> candidates,
            Map<S, Double> emissionLogProbabilities, 
            Map<Transition<S>, Double> transitionLogProbabilities, 
            Map<Transition<S>, D> transitionDescriptors) {
        if (message == null) {
            throw new IllegalStateException(
                    "startWithInitialStateProbabilities() or startWithInitialObservation() "
                    + "must be called first.");
        }
        if (isBroken) {
            throw new IllegalStateException("Method must not be called after an HMM break.");
        }
        
        // Forward step
        ForwardStepResult<S, O, D> forwardStepResult = forwardStep(observation, prevCandidates, 
                candidates, message, emissionLogProbabilities, transitionLogProbabilities, 
                transitionDescriptors);
        isBroken = hmmBreak(forwardStepResult.message);
        if (isBroken) return;
        if (messageHistory != null) {
            messageHistory.add(forwardStepResult.message);
        }
        message = forwardStepResult.message;
        lastExtendedStates = forwardStepResult.backPointers;

        prevCandidates = candidates;
    }
    
    /**
     * See {@link ViterbiAlgorithm#nextStep(Object, Collection, Map, Map, Map)}
     */
    public void nextStep(O observation, Collection<S> candidates,
            Map<S, Double> emissionLogProbabilities, 
            Map<Transition<S>, Double> transitionLogProbabilities) {
        nextStep(observation, candidates, emissionLogProbabilities, transitionLogProbabilities, 
                new LinkedHashMap<Transition<S>, D>());
    }
    
    /**
     * Returns the most likely sequence of states for all time steps.
     * Formally, this is argmax p(s_1, ..., s_T | o_1, ..., o_T) with respect to s_1, ..., s_T,
     * where s_t is a state candidate at time step t, o_t is the observation at time step t and T is
     * the number of time steps.
     * 
     * If an HMM break occurred in the last time step t, then the most likely sequence up to t-1 is 
     * returned.
     * See {@link ViterbiAlgorithm#isBroken()}
     */
    public List<SequenceState<S, O, D>> computeMostLikelySequence() {
        if (message == null) {
            // Return empty most likely sequence if there are no time steps or if initial 
            // observations caused an HMM break.
            return new ArrayList<>();  
        } else {
            assert(!message.isEmpty()); // Otherwise an HMM break occurred and message would be null
            return retrieveMostLikelySequence(mostLikelyState(message));
        }
    }
    
    /**
     * Returns whether an HMM occurred in the last time step.
     * 
     * An HMM break means that the probability of all states equals zero.
     */
    public boolean isBroken() {
        return isBroken;
    }
    
    /**
     *  Sequence of computed messages for each time step. Is null if message history
     *  is not kept (see compute()).
     *
     *  For each state s_t of the time step t, messageHistory.get(t).get(s_t) contains the log
     *  probability of the most likely sequence ending in state s_t with given observations
     *  o_1, ..., o_t.
     *  Formally, this is max log p(s_1, ..., s_t, o_1, ..., o_t) w.r.t. s_1, ..., s_{t-1}.
     *  Note that to compute the most likely state sequence, it is sufficient and more
     *  efficient to compute in each time step the joint probability of states and observations
     *  instead of computing the conditional probability of states given the observations.
     */
    List<Map<S, Double>> messageHistory() {
        return messageHistory;
    }
    
    String messageHistoryString() {
        StringBuffer sb = new StringBuffer();
        sb.append("Message history with log probabilies\n\n");
        int i = 0;
        for (Map<S, Double> message : messageHistory) {
            sb.append("Time step " + i + "\n");
            i++;
            for (S state : message.keySet()) {
                sb.append(state + ": " + message.get(state) + "\n");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
    
    /**
     * Each back pointer from a state s contains the previous state of the most
     * likely state sequence passing at time step t through state s.
     */
    Map<S, ExtendedState<S, O, D>> backPointers() {
        return lastExtendedStates;
    }
    
    /**
     * Returns whether the specified message is either empty or only contains state candidates
     * with 0 probability and thus causes the HMM to break.
     */
    private boolean hmmBreak(Map<S, Double> message) {
        for (double logProbability : message.values()) {
            if (logProbability != Double.NEGATIVE_INFINITY) {
                return false;
            }
        }
        return true;
    }

    /**
     * @param observation Use only if HMM only starts with first observation.
     */
    private void initializeStateProbabilities(O observation, Collection<S> candidates, 
            Map<S, Double> initialLogProbabilities) {
        if (message != null) {
            throw new IllegalStateException("Initial probabilities have already been set.");
        }

        // Set initial log probability for each start state candidate based on first observation.
        // Do not assign initialLogProbabilities directly to message to not rely on its iteration
        // order.
        final Map<S, Double> initialMessage = new LinkedHashMap<>();
        for (S candidate : candidates) {
            final Double logProbability = initialLogProbabilities.get(candidate);
            if (logProbability == null) {
                throw new NullPointerException("No initial probability for " + candidate);
            }
            initialMessage.put(candidate, logProbability);
        }
        
        isBroken = hmmBreak(initialMessage);
        if (isBroken) return;
        
        message = initialMessage;
        if (messageHistory != null) {
            messageHistory.add(message);
        }
        
        lastExtendedStates = new LinkedHashMap<>();
        for (S candidate : candidates) {
            lastExtendedStates.put(candidate, 
                    new ExtendedState<S, O, D>(candidate, null, observation, null));
        }
        
        prevCandidates = candidates;           
    }    
    
    /**
     * Computes the new forward message and the back pointers to the previous states (next entry of
     * backPointerSequence). 
     * 
     * @throws NullPointerException if any emission probability is missing.
     */
    private ForwardStepResult<S, O, D> forwardStep(O observation, Collection<S> prevCandidates, 
            Collection<S> curCandidates, Map<S, Double> message, 
            Map<S, Double> emissionLogProbabilities, 
            Map<Transition<S>, Double> transitionLogProbabilities, 
            Map<Transition<S>,D> transitionDescriptors) {
        final ForwardStepResult<S, O, D> result = new ForwardStepResult<>(curCandidates.size());
        assert( !prevCandidates.isEmpty());

        for (S curState : curCandidates) {
            double maxLogProbability = Double.NEGATIVE_INFINITY;
            S maxPrevState = null;
            for (S prevState : prevCandidates) {
                final double logProbability = message.get(prevState) + transitionLogProbability(
                        transitionLogProbabilities, prevState, curState);
                if (logProbability > maxLogProbability) {
                    maxLogProbability = logProbability;
                    maxPrevState = prevState;
                }
            }
            // Throws NullPointerException if curState is not stored in the map.
            result.message.put(curState, maxLogProbability 
                    + emissionLogProbabilities.get(curState));

            // Note that maxPrevState == null if there is no transition with non-zero probability.
            // In this case curState has zero probability and will not be part of the most likely
            // sequence, so we don't need an ExtendedState.
            if (maxPrevState != null) {
                final Transition<S> transition = new Transition<>(maxPrevState, curState);
                final ExtendedState<S, O, D> extendedState = new ExtendedState<>(curState, 
                        lastExtendedStates.get(maxPrevState), observation, 
                        transitionDescriptors.get(transition));
                result.backPointers.put(curState, extendedState);
            }
        }
        return result;
    }

    private double transitionLogProbability(Map<Transition<S>, Double> transitionLogProbabilities,
            S prevState, S curState) {
        final Double transitionLogProbability = 
                transitionLogProbabilities.get(new Transition<S>(prevState, curState));
        if (transitionLogProbability == null) {
            return Double.NEGATIVE_INFINITY; // Transition has zero probability.
        } else {
            return transitionLogProbability;
        }
    }

    /**
     * Retrieves a state with maximum probability.
     */
    private S mostLikelyState(Map<S, Double> message) {
        if (message.isEmpty()) {
            throw new IllegalArgumentException("message must not be empty.");
        }
        
        // Set first state as most likely state.
        final Iterator<Map.Entry<S, Double>> entryIter = message.entrySet().iterator();
        final Map.Entry<S, Double> firstEntry = entryIter.next();
        S result = firstEntry.getKey();
        double maxLogProbability = firstEntry.getValue();

        // Check remaining states.
        while (entryIter.hasNext()) {
            final Map.Entry<S, Double> entry = entryIter.next();
            if (entry.getValue() > maxLogProbability) {
                maxLogProbability = entry.getValue();
                result = entry.getKey();
            }
        }
        return result;
    }

    /**
     * Retrieves most likely sequence from the internal back pointer sequence ending in the 
     * specified last state.
     */
    private List<SequenceState<S, O, D>> retrieveMostLikelySequence(S lastState) {
        final List<SequenceState<S, O, D>> result = new ArrayList<>();

        // Retrieve most likely state sequence in reverse order
        ExtendedState<S, O, D> es = lastExtendedStates.get(lastState);
        while(es != null) {
            final SequenceState<S, O, D> ss = new SequenceState<>(es.state, es.observation, 
                    es.transitionDescriptor);
            result.add(ss);
            es = es.backPointer;
        }
        
        Collections.reverse(result);
        return result;
    }


}
