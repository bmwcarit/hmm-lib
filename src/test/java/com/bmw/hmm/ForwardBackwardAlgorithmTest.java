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

import static org.junit.Assert.assertEquals;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import com.bmw.hmm.Transition;

public class ForwardBackwardAlgorithmTest {

    private static class Rain {
        final static Rain T = new Rain();
        final static Rain F = new Rain();

        @Override
        public String toString() {
            if (this == T) {
                return "Rain";
            } else if (this == F) {
                return "Sun";
            }
            throw new IllegalStateException();
        }
    }

    private static class Umbrella {
        final static Umbrella T = new Umbrella();
        final static Umbrella F = new Umbrella();

        @Override
        public String toString() {
            if (this == T) {
                return "Umbrella";
            } else if (this == F) {
                return "No umbrella";
            }
            throw new IllegalStateException();
        }
    }

    /**
     * Example taken from https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm.
     */
    @Test
    public void testForwardBackward() {
        final List<Rain> candidates = new ArrayList<>();
        candidates.add(Rain.T);
        candidates.add(Rain.F);

        final Map<Rain, Double> initialStateProbabilities = new LinkedHashMap<>();
        initialStateProbabilities.put(Rain.T, 0.5);
        initialStateProbabilities.put(Rain.F, 0.5);

        final Map<Rain, Double> emissionProbabilitiesForUmbrella = new LinkedHashMap<>();
        emissionProbabilitiesForUmbrella.put(Rain.T, 0.9);
        emissionProbabilitiesForUmbrella.put(Rain.F, 0.2);

        final Map<Rain, Double> emissionProbabilitiesForNoUmbrella = new LinkedHashMap<>();
        emissionProbabilitiesForNoUmbrella.put(Rain.T, 0.1);
        emissionProbabilitiesForNoUmbrella.put(Rain.F, 0.8);

        final Map<Transition<Rain>, Double> transitionProbabilities = new LinkedHashMap<>();
        transitionProbabilities.put(new Transition<Rain>(Rain.T, Rain.T), 0.7);
        transitionProbabilities.put(new Transition<Rain>(Rain.T, Rain.F), 0.3);
        transitionProbabilities.put(new Transition<Rain>(Rain.F, Rain.T), 0.3);
        transitionProbabilities.put(new Transition<Rain>(Rain.F, Rain.F), 0.7);

        final ForwardBackwardAlgorithm<Rain, Umbrella> fw = new ForwardBackwardAlgorithm<>();
        fw.startWithInitialStateProbabilities(candidates, initialStateProbabilities);
        fw.nextStep(Umbrella.T, candidates, emissionProbabilitiesForUmbrella,
                transitionProbabilities);
        fw.nextStep(Umbrella.T, candidates, emissionProbabilitiesForUmbrella,
                transitionProbabilities);
        fw.nextStep(Umbrella.F, candidates, emissionProbabilitiesForNoUmbrella,
                transitionProbabilities);
        fw.nextStep(Umbrella.T, candidates, emissionProbabilitiesForUmbrella,
                transitionProbabilities);
        fw.nextStep(Umbrella.T, candidates, emissionProbabilitiesForUmbrella,
                transitionProbabilities);

        final List<Map<Rain, Double>> result = fw.computeSmoothingProbabilities();
        assertEquals(6, result.size());
        final double DELTA = 1e-4;
        assertEquals(0.6469, result.get(0).get(Rain.T), DELTA);
        assertEquals(0.3531, result.get(0).get(Rain.F), DELTA);
        assertEquals(0.8673, result.get(1).get(Rain.T), DELTA);
        assertEquals(0.1327, result.get(1).get(Rain.F), DELTA);
        assertEquals(0.8204, result.get(2).get(Rain.T), DELTA);
        assertEquals(0.1796, result.get(2).get(Rain.F), DELTA);
        assertEquals(0.3075, result.get(3).get(Rain.T), DELTA);
        assertEquals(0.6925, result.get(3).get(Rain.F), DELTA);
        assertEquals(0.8204, result.get(4).get(Rain.T), DELTA);
        assertEquals(0.1796, result.get(4).get(Rain.F), DELTA);
        assertEquals(0.8673, result.get(5).get(Rain.T), DELTA);
        assertEquals(0.1327, result.get(5).get(Rain.F), DELTA);
    }



}