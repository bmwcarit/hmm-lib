/**
 * Copyright (C) 2015, BMW Car IT GmbH
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

package de.bmw.hmm;

import java.util.Iterator;
import java.util.List;

/**
 * Implements an HMM for time-inhomogeneous Markov processes, meaning that the set of states and
 * state transition probabilities are not fixed for all time steps. Hence, this library requires
 * that the user provides the state candidates for each time step and the computation of state
 * transitions.
 *
 * An initial probability for each starting state does not need to be provided. Instead, the initial
 * state probability for a state s given the initial observation o is computed as p(o|s).
 *
 * @param <S> state class/interface
 * @param <O> observation class/interface
 */
public class Hmm<S, O> {

    private final HmmProbabilities<S, O> hmmProbabilities;

    public Hmm(HmmProbabilities<S, O> hmmProbabilities) {
        if (hmmProbabilities == null) {
            throw new NullPointerException("hmmProbabilities must not be null.");
        }

        this.hmmProbabilities = hmmProbabilities;
    }

    /**
     * Computes the most likely sequence of states given the specified time step sequence.
     *
     * Formally, this is argmax p(s_1, ..., s_T | o_1, ..., o_T) with respect to s_1, ..., s_T,
     * where s_t is a state candidate at time step t, o_t is the observation at time step t and T is
     * the number of time steps.
     * @param timeStepIter Iterates the finite sequence of time steps. Allows to compute candidates
     * and observations on the fly by implementing the Iterator interface.
     */
    public List<S> computeMostLikelySequence(Iterator<TimeStep<S, O>> timeStepIter) {
        ViterbiAlgorithm<S, O> viterbi = new ViterbiAlgorithm<>();
        return viterbi.compute(hmmProbabilities, timeStepIter, false).
                mostLikelySequence;
    }

}
