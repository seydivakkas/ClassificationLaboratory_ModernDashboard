namespace ClassificationApp;

public sealed class MetricCard : RoundedCard
{
    private readonly Label _value;
    private readonly Label _caption;

    public string Value
    {
        get => _value.Text;
        set => _value.Text = value;
    }

    public MetricCard(string caption, string value = "-")
    {
        Height = 78;
        Padding = new Padding(12, 8, 12, 8);
        _caption = UiTheme.Label(caption.ToUpperInvariant(), 7.5f, true, UiTheme.Muted);
        _value = UiTheme.Label(value, 16f, true, UiTheme.Text);
        _caption.Location = new Point(12, 10);
        _value.Location = new Point(12, 32);
        Controls.Add(_caption);
        Controls.Add(_value);
    }
}
