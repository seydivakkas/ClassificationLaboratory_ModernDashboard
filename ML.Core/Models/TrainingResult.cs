namespace ML.Core.Models;

public sealed class TrainingResult
{
    public int EpochsRun { get; init; }
    public double FinalLoss { get; init; }
    public double Accuracy { get; init; }
    public IReadOnlyList<double> ErrorHistory { get; init; } = Array.Empty<double>();
}
