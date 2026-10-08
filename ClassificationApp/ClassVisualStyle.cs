namespace ClassificationApp;

public enum MarkerShape
{
    Circle,
    Square,
    Triangle,
    Diamond,
    Cross
}

public sealed class ClassVisualStyle
{
    public Color Color { get; set; }
    public MarkerShape Marker { get; set; }

    public ClassVisualStyle(Color color, MarkerShape marker)
    {
        Color = color;
        Marker = marker;
    }
}
