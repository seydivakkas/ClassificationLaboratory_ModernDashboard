using ML.Core.Models;

namespace ML.Core.Classification;

public interface IProgressiveClassifier : IClassifier
{
    TrainingResult FitWithProgress(
        IReadOnlyList<double[]> samples,
        IReadOnlyList<int> labels,
        Action<ClassifierEpochSnapshot>? onSnapshot,
        int snapshotEvery = 1);
}
