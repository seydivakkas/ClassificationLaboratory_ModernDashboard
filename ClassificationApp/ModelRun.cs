using ML.Core.Classification;
using ML.Core.Models;

namespace ClassificationApp;

public sealed class ModelRun
{
    public required int Id { get; init; }
    public required string Algorithm { get; init; }
    public required IClassifier Classifier { get; init; }
    public required TrainingResult Training { get; init; }
    public required ClassificationEvaluation Evaluation { get; init; }
    public required int DatasetVersion { get; init; }
    public required int TestPercent { get; init; }
    public required SamplePoint[] Misclassified { get; init; }
}
