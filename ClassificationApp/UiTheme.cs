using System.Drawing.Drawing2D;

namespace ClassificationApp;

public static class UiTheme
{
    public static readonly Color Window = Color.FromArgb(8, 18, 38);
    public static readonly Color Sidebar = Color.FromArgb(11, 27, 55);
    public static readonly Color Surface = Color.FromArgb(15, 35, 68);
    public static readonly Color Surface2 = Color.FromArgb(18, 43, 82);
    public static readonly Color Plot = Color.FromArgb(7, 20, 42);
    public static readonly Color Accent = Color.FromArgb(41, 121, 255);
    public static readonly Color AccentSoft = Color.FromArgb(27, 79, 150);
    public static readonly Color Cyan = Color.FromArgb(39, 203, 255);
    public static readonly Color Text = Color.FromArgb(235, 244, 255);
    public static readonly Color Muted = Color.FromArgb(150, 172, 205);
    public static readonly Color Border = Color.FromArgb(42, 67, 103);
    public static readonly Color Success = Color.FromArgb(47, 201, 140);
    public static readonly Color Warning = Color.FromArgb(255, 193, 7);
    public static readonly Color Danger = Color.FromArgb(255, 82, 99);

    public static Button Button(string text, bool primary = false)
    {
        var b = new Button
        {
            Text = text,
            Height = 38,
            AutoSize = false,
            FlatStyle = FlatStyle.Flat,
            BackColor = primary ? Accent : Surface2,
            ForeColor = Text,
            Cursor = Cursors.Hand,
            Font = new Font("Segoe UI Semibold", 9.5f),
            Margin = new Padding(4)
        };
        b.FlatAppearance.BorderSize = 1;
        b.FlatAppearance.BorderColor = primary ? Accent : Border;
        b.FlatAppearance.MouseOverBackColor = primary ? Color.FromArgb(62, 137, 255) : Color.FromArgb(26, 55, 98);
        return b;
    }

    public static Label Label(string text, float size = 9f, bool bold = false, Color? color = null)
        => new()
        {
            Text = text,
            AutoSize = true,
            ForeColor = color ?? Text,
            Font = new Font("Segoe UI", size, bold ? FontStyle.Bold : FontStyle.Regular),
            BackColor = Color.Transparent
        };

    public static Panel Card(int padding = 12)
        => new()
        {
            BackColor = Surface,
            Padding = new Padding(padding),
            Margin = new Padding(6)
        };

    public static void StyleNumeric(NumericUpDown n)
    {
        n.BackColor = Surface2;
        n.ForeColor = Text;
        n.BorderStyle = BorderStyle.FixedSingle;
        n.Font = new Font("Segoe UI", 9.5f);
        n.Height = 30;
    }

    public static void StyleCombo(ComboBox c)
    {
        c.BackColor = Surface2;
        c.ForeColor = Text;
        c.FlatStyle = FlatStyle.Flat;
        c.Font = new Font("Segoe UI", 9.5f);
    }
}

public sealed class RoundedCard : Panel
{
    public int Radius { get; set; } = 12;
    public Color BorderColor { get; set; } = UiTheme.Border;

    public RoundedCard()
    {
        DoubleBuffered = true;
        BackColor = UiTheme.Surface;
        Padding = new Padding(12);
        Margin = new Padding(6);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        if (Width < 4 || Height < 4) return;
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        using var path = CreateRoundRect(new Rectangle(0, 0, Width - 1, Height - 1), Radius);
        using var pen = new Pen(BorderColor);
        e.Graphics.DrawPath(pen, path);
    }

    private static GraphicsPath CreateRoundRect(Rectangle r, int radius)
    {
        int d = Math.Max(2, radius * 2);
        var path = new GraphicsPath();
        path.AddArc(r.X, r.Y, d, d, 180, 90);
        path.AddArc(r.Right - d, r.Y, d, d, 270, 90);
        path.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
        path.AddArc(r.X, r.Bottom - d, d, d, 90, 90);
        path.CloseFigure();
        return path;
    }
}
