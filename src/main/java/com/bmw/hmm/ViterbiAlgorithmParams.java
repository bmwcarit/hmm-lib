package com.bmw.hmm;

/**
 * Parameters for {@link ViterbiAlgorithm}.
 */
public class ViterbiAlgorithmParams {

    private boolean keepMessageHistory = false;
    private boolean computeSmoothingProbabilities = false;

    /**
     * Whether to store intermediate forward messages
     * (probabilities of intermediate most likely paths) for debugging.
     */
    public ViterbiAlgorithmParams setKeepMessageHistory(boolean value) {
        this.keepMessageHistory = value;
        return this;
    }

    /**
     * Whether to compute smoothing probabilities using the {@link ForwardBackwardAlgorithm}
     * for the states of the most likely sequence. Note that this significantly increases
     * computation time and memory footprint.
     */
    public ViterbiAlgorithmParams setComputeSmoothingProbabilities(boolean value) {
        this.computeSmoothingProbabilities = value;
        return this;
    }

    public boolean isKeepMessageHistory() {
        return keepMessageHistory;
    }

    public boolean isComputeSmoothingProbabilities() {
        return computeSmoothingProbabilities;
    }

}
