namespace ClassificationApp;

public sealed class ConfusionMatrixPanel : Panel
{
    private int[,]? _matrix;
    public int[,]? Matrix { get => _matrix; set { _matrix = value; Invalidate(); } }

    public ConfusionMatrixPanel()
    {
        DoubleBuffered = true;
        BackColor = UiTheme.Surface;
        Padding = new Padding(6);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        var g = e.Graphics;
        g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
        using var title = new Font("Segoe UI Semibold", 10f);
        using var small = new Font("Segoe UI", 8f);
        using var text = new SolidBrush(UiTheme.Text);
        using var muted = new SolidBrush(UiTheme.Muted);
        g.DrawString("Confusion Matrix", title, text, 10, 8);

        if (_matrix is null)
        {
            g.DrawString("Model eğitildikten sonra test sonuçları burada görünür.", small, muted, 10, 36);
            return;
        }

        int n = _matrix.GetLength(0);
        int left = 42, top = 48;
        int cell = Math.Max(28, Math.Min((Width - left - 15) / Math.Max(1, n), (Height - top - 20) / Math.Max(1, n)));
        int max = 1;
        foreach (int v in _matrix) max = Math.Max(max, v);

        for (int c = 0; c < n; c++)
        {
            string s = c.ToString();
            g.DrawString(s, small, muted, left + c * cell + cell / 2 - 4, top - 18);
            g.DrawString(s, small, muted, 20, top + c * cell + cell / 2 - 7);
        }

        using var border = new Pen(UiTheme.Border);
        for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
        {
            int value = _matrix[r, c];
            double intensity = (double)value / max;
            Color baseColor = r == c ? UiTheme.Success : UiTheme.Danger;
            using var fill = new SolidBrush(Color.FromArgb(45 + (int)(145 * intensity), baseColor));
            var rect = new Rectangle(left + c * cell, top + r * cell, cell - 2, cell - 2);
            g.FillRectangle(fill, rect);
            g.DrawRectangle(border, rect);
            string label = value.ToString();
            var size = g.MeasureString(label, small);
            g.DrawString(label, small, text, rect.X + (rect.Width - size.Width) / 2, rect.Y + (rect.Height - size.Height) / 2);
        }

        g.DrawString("Tahmin →", small, muted, left, top + n * cell + 2);
        g.DrawString("Gerçek", small, muted, 2, top - 1);
    }
}
