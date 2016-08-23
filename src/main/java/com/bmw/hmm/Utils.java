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

package com.bmw.hmm;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Implementation utilities.
 */
class Utils {

    static int initialHashMapCapacity(int maxElements) {
        // Default load factor of HashMaps is 0.75
        return (int)(maxElements / 0.75) + 1;
    }

    static <S> Map<S, Double> logToNonLogProbabilities(Map<S, Double> logProbabilities) {
        final Map<S, Double> result = new LinkedHashMap<>();
        for (Map.Entry<S, Double> entry : logProbabilities.entrySet()) {
            result.put(entry.getKey(), Math.exp(entry.getValue()));
        }
        return result;
    }

    /**
     * Note that this check must not be used for probability densities.
     */
    static boolean probabilityInRange(double probability, double delta) {
        return probability >= -delta && probability <= 1.0 + delta;
    }

}
