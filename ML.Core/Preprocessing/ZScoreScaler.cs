namespace ML.Core.Preprocessing;

public sealed class ZScoreScaler
{
    public double[] Mean { get; private set; } = Array.Empty<double>();
    public double[] Std { get; private set; } = Array.Empty<double>();

    public void Fit(IReadOnlyList<double[]> samples)
    {
        if (samples.Count == 0)
            throw new ArgumentException("En az bir örnek gereklidir.", nameof(samples));

        int dim = samples[0].Length;
        if (dim == 0)
            throw new ArgumentException("Örnek boyutu sıfır olamaz.", nameof(samples));

        Mean = new double[dim];
        Std = new double[dim];

        foreach (var sample in samples)
        {
            if (sample.Length != dim)
                throw new ArgumentException("Tüm örneklerin boyutu aynı olmalıdır.", nameof(samples));

            for (int d = 0; d < dim; d++)
                Mean[d] += sample[d];
        }

        for (int d = 0; d < dim; d++)
            Mean[d] /= samples.Count;

        foreach (var sample in samples)
        {
            for (int d = 0; d < dim; d++)
            {
                double diff = sample[d] - Mean[d];
                Std[d] += diff * diff;
            }
        }

        for (int d = 0; d < dim; d++)
        {
            Std[d] = Math.Sqrt(Std[d] / samples.Count);
            if (Std[d] < 1e-12)
                Std[d] = 1.0;
        }
    }

    public double[] Transform(IReadOnlyList<double> sample)
    {
        if (Mean.Length == 0)
            throw new InvalidOperationException("Scaler önce Fit edilmelidir.");
        if (sample.Count != Mean.Length)
            throw new ArgumentException("Örnek boyutu scaler boyutuyla eşleşmiyor.", nameof(sample));

        var result = new double[Mean.Length];
        for (int d = 0; d < Mean.Length; d++)
            result[d] = (sample[d] - Mean[d]) / Std[d];
        return result;
    }

    public double[][] Transform(IReadOnlyList<double[]> samples)
    {
        var result = new double[samples.Count][];
        for (int i = 0; i < samples.Count; i++)
            result[i] = Transform(samples[i]);
        return result;
    }
}
