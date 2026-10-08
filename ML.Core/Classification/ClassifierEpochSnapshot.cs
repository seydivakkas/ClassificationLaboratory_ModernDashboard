namespace ML.Core.Classification;

public sealed class ClassifierEpochSnapshot
{
    public required int Epoch { get; init; }
    public required double Loss { get; init; }
    public required IClassifier Model { get; init; }
}
