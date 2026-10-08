namespace ML.Core.Classification;

public sealed class ClassificationEvaluation
{
    public required int[,] ConfusionMatrix { get; init; }
    public required int[] Supports { get; init; }
    public double Accuracy { get; init; }
    public double MacroPrecision { get; init; }
    public double MacroRecall { get; init; }
    public double MacroF1 { get; init; }
    public int Correct { get; init; }
    public int Total { get; init; }
}

public static class ClassificationEvaluator
{
    public static ClassificationEvaluation Evaluate(
        IClassifier classifier,
        IReadOnlyList<double[]> samples,
        IReadOnlyList<int> labels,
        int classCount)
    {
        if (samples.Count != labels.Count)
            throw new ArgumentException("Örnek ve etiket sayıları eşit olmalıdır.");
        if (classCount < 2)
            throw new ArgumentOutOfRangeException(nameof(classCount));

        var matrix = new int[classCount, classCount];
        var supports = new int[classCount];
        int correct = 0;

        for (int i = 0; i < samples.Count; i++)
        {
            int expected = labels[i];
            int predicted = classifier.Predict(samples[i]);
            if (expected < 0 || expected >= classCount) continue;
            predicted = Math.Clamp(predicted, 0, classCount - 1);
            matrix[expected, predicted]++;
            supports[expected]++;
            if (expected == predicted) correct++;
        }

        double sumPrecision = 0, sumRecall = 0, sumF1 = 0;
        int activeClasses = 0;
        for (int c = 0; c < classCount; c++)
        {
            int tp = matrix[c, c];
            int fp = 0;
            int fn = 0;
            for (int r = 0; r < classCount; r++) if (r != c) fp += matrix[r, c];
            for (int p = 0; p < classCount; p++) if (p != c) fn += matrix[c, p];

            if (supports[c] == 0) continue;
            activeClasses++;
            double precision = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
            double recall = tp + fn == 0 ? 0 : (double)tp / (tp + fn);
            double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
            sumPrecision += precision;
            sumRecall += recall;
            sumF1 += f1;
        }

        int denom = Math.Max(1, activeClasses);
        return new ClassificationEvaluation
        {
            ConfusionMatrix = matrix,
            Supports = supports,
            Accuracy = samples.Count == 0 ? 0 : (double)correct / samples.Count,
            MacroPrecision = sumPrecision / denom,
            MacroRecall = sumRecall / denom,
            MacroF1 = sumF1 / denom,
            Correct = correct,
            Total = samples.Count
        };
    }
}
