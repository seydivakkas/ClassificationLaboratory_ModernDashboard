using ML.Core.Models;

namespace ML.Core.Classification;

public interface IClassifier
{
    TrainingResult Fit(IReadOnlyList<double[]> samples, IReadOnlyList<int> labels);
    int Predict(IReadOnlyList<double> sample);
    double[] PredictScores(IReadOnlyList<double> sample);
}
