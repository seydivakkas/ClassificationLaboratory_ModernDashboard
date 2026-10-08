using ML.Core.Models;

namespace ML.Core.Classification;

internal sealed class FrozenClassifier : IClassifier
{
    private readonly Func<IReadOnlyList<double>, double[]> _scoreFunction;

    public FrozenClassifier(Func<IReadOnlyList<double>, double[]> scoreFunction)
    {
        _scoreFunction = scoreFunction;
    }

    public TrainingResult Fit(IReadOnlyList<double[]> samples, IReadOnlyList<int> labels) =>
        throw new NotSupportedException("Bu nesne eğitim anlık görüntüsüdür ve yeniden eğitilemez.");

    public double[] PredictScores(IReadOnlyList<double> sample) => _scoreFunction(sample);

    public int Predict(IReadOnlyList<double> sample)
    {
        var scores = PredictScores(sample);
        int best = 0;
        for (int i = 1; i < scores.Length; i++)
            if (scores[i] > scores[best]) best = i;
        return best;
    }
}
