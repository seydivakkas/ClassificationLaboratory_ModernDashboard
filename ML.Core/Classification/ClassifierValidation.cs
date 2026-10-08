namespace ML.Core.Classification;

internal static class ClassifierValidation
{
    public static int Validate(IReadOnlyList<double[]> samples, IReadOnlyList<int> labels, int minimumClasses = 2)
    {
        if (samples.Count == 0)
            throw new ArgumentException("En az bir eğitim örneği gereklidir.", nameof(samples));
        if (samples.Count != labels.Count)
            throw new ArgumentException("Örnek ve etiket sayıları eşit olmalıdır.");

        int dim = samples[0].Length;
        if (dim == 0)
            throw new ArgumentException("Girdi boyutu sıfır olamaz.", nameof(samples));

        foreach (var x in samples)
            if (x.Length != dim)
                throw new ArgumentException("Tüm örneklerin girdi boyutu aynı olmalıdır.", nameof(samples));

        int classCount = labels.Distinct().Count();
        if (classCount < minimumClasses)
            throw new ArgumentException($"En az {minimumClasses} farklı sınıf gereklidir.", nameof(labels));

        int maxLabel = labels.Max();
        int minLabel = labels.Min();
        if (minLabel < 0 || labels.Distinct().OrderBy(x => x).SequenceEqual(Enumerable.Range(0, maxLabel + 1)) == false)
            throw new ArgumentException("Sınıf etiketleri 0'dan başlayıp ardışık olmalıdır.", nameof(labels));

        return dim;
    }

    public static double Accuracy(IClassifier classifier, IReadOnlyList<double[]> samples, IReadOnlyList<int> labels)
    {
        int correct = 0;
        for (int i = 0; i < samples.Count; i++)
            if (classifier.Predict(samples[i]) == labels[i])
                correct++;
        return (double)correct / samples.Count;
    }
}
