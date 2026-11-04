using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace LangSwapDesktop
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainWindow());
        }
    }

    public sealed class MainWindow : Form
    {
        private static readonly Color Ink = Color.FromArgb(11, 16, 29);
        private static readonly Color Surface = Color.FromArgb(21, 29, 47);
        private static readonly Color Raised = Color.FromArgb(29, 39, 61);
        private static readonly Color Border = Color.FromArgb(54, 66, 94);
        private static readonly Color TextColor = Color.FromArgb(242, 244, 251);
        private static readonly Color Muted = Color.FromArgb(166, 177, 198);
        private static readonly Color Accent = Color.FromArgb(165, 142, 249);
        private static readonly Color Mint = Color.FromArgb(103, 223, 187);

        private readonly TextBox url = new TextBox();
        private readonly TextBox output = new TextBox();
        private readonly TextBox log = new TextBox();
        private readonly ComboBox sourceLanguage = new ComboBox();
        private readonly ComboBox targetLanguage = new ComboBox();
        private readonly ComboBox mode = new ComboBox();
        private readonly Label sourceFileName = new Label();
        private readonly Label status = new Label();
        private readonly Button startButton = new Button();
        private readonly Button cancelButton = new Button();
        private readonly Button watchButton = new Button();
        private readonly ListView recent = new ListView();
        private readonly ProgressBar progress = new ProgressBar();
        private readonly Panel formPage = new Panel();
        private readonly Panel libraryPage = new Panel();
        private readonly Button createTab = new Button();
        private readonly Button libraryTab = new Button();

        private string selectedFile;
        private string lastResultPath;
        private string currentContainer;
        private volatile Process currentProcess;
        private bool imageReady;
        private bool qwenImageReady;
        private bool busy;
        private readonly string backendRoot;
        private readonly string modelDirectory;
        private readonly string dataDirectory;
        private readonly string defaultOutput;
        private const string BaseImage = "langswap-french-desktop:0.1.0";
        private const string QwenImage = "langswap-qwen-desktop:0.1.0";

        public MainWindow()
        {
            Text = "LangSwap — application Windows";
            Icon = MakeIcon();
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size(1120, 720);
            MinimumSize = new Size(860, 610);
            BackColor = Ink;
            ForeColor = TextColor;
            Font = new Font("Segoe UI", 10f);
            AutoScaleMode = AutoScaleMode.Dpi;

            string bundled = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "runtime", "langswap");
            string checkout = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                "..", "..", "runtime", "langswap"));
            backendRoot = Directory.Exists(bundled) ? bundled : checkout;
            string appData = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LangSwap");
            modelDirectory = FolderFromEnvironment("LANGSWAP_MODELS_DIR", Path.Combine(appData, "models"));
            dataDirectory = FolderFromEnvironment("LANGSWAP_JOBS_DIR", Path.Combine(appData, "jobs"));
            defaultOutput = FolderFromEnvironment("LANGSWAP_OUTPUT_DIR",
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyVideos), "LangSwap"));

            BuildLayout();
            output.Text = defaultOutput;
            sourceLanguage.SelectedIndex = 0;
            targetLanguage.SelectedIndex = 0;
            mode.SelectedIndex = 0;
            RefreshRecent();
            ShowPage(false);
        }

        private static string FolderFromEnvironment(string name, string fallback)
        {
            string value = Environment.GetEnvironmentVariable(name);
            return Path.GetFullPath(Environment.ExpandEnvironmentVariables(
                String.IsNullOrWhiteSpace(value) ? fallback : value));
        }

        private static Label Label(string text, int x, int y, int width, int height, float size, Color color, bool bold)
        {
            return new Label {
                Text = text, Location = new Point(x, y), Size = new Size(width, height),
                Font = new Font("Segoe UI", size, bold ? FontStyle.Bold : FontStyle.Regular),
                ForeColor = color, BackColor = Color.Transparent,
                AutoEllipsis = true
            };
        }

        private static Button Button(string text, Color background, Color foreground)
        {
            Button button = new Button {
                Text = text, FlatStyle = FlatStyle.Flat, BackColor = background,
                ForeColor = foreground, Cursor = Cursors.Hand,
                Font = new Font("Segoe UI", 10f, FontStyle.Bold)
            };
            button.FlatAppearance.BorderSize = 0;
            button.FlatAppearance.MouseOverBackColor = ControlPaint.Light(background, 0.15f);
            return button;
        }

        private static void StyleInput(Control control)
        {
            control.BackColor = Raised;
            control.ForeColor = TextColor;
            control.Font = new Font("Segoe UI", 10.5f);
        }

        private static Panel Card(int y, int height)
        {
            Panel card = new Panel { Location = new Point(0, y), Size = new Size(820, height),
                                     Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top,
                                     BackColor = Surface };
            card.Paint += (sender, args) => {
                using (Pen pen = new Pen(Border)) {
                    args.Graphics.DrawRectangle(pen, 0, 0, card.ClientSize.Width - 1, card.ClientSize.Height - 1);
                }
            };
            return card;
        }

        private void BuildLayout()
        {
            TableLayoutPanel shell = new TableLayoutPanel {
                Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1,
                BackColor = Ink, Margin = Padding.Empty, Padding = Padding.Empty
            };
            shell.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 226));
            shell.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            shell.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            Controls.Add(shell);
            Panel side = new Panel { Dock = DockStyle.Fill, BackColor = Color.FromArgb(16, 23, 39) };
            shell.Controls.Add(side, 0, 0);
            Panel mark = new Panel { Location = new Point(24, 25), Size = new Size(42, 42) };
            mark.Paint += PaintMark;
            side.Controls.Add(mark);
            side.Controls.Add(Label("LangSwap", 76, 24, 135, 30, 18, TextColor, true));
            side.Controls.Add(Label("STUDIO LOCAL", 77, 52, 130, 18, 7.5f, Accent, true));

            side.Controls.Add(Label("ATELIER", 24, 128, 160, 20, 8, Muted, true));
            createTab.Text = "Créer une vidéo";
            libraryTab.Text = "Mes versions";
            foreach (Button tab in new[] { createTab, libraryTab }) {
                tab.FlatStyle = FlatStyle.Flat;
                tab.FlatAppearance.BorderSize = 0;
                tab.TextAlign = ContentAlignment.MiddleLeft;
                tab.ForeColor = TextColor;
                tab.BackColor = Color.Transparent;
                tab.Size = new Size(202, 43);
                tab.Location = new Point(12, tab == createTab ? 154 : 202);
                tab.Font = new Font("Segoe UI", 10, FontStyle.Regular);
                tab.Cursor = Cursors.Hand;
                side.Controls.Add(tab);
            }
            createTab.Click += (sender, args) => ShowPage(false);
            libraryTab.Click += (sender, args) => { RefreshRecent(); ShowPage(true); };
            Label bottom = Label("GRATUIT  ·  WINDOWS  ·  LOCAL", 24, 0, 188, 32, 8, Mint, true);
            bottom.Anchor = AnchorStyles.Bottom | AnchorStyles.Left;
            bottom.Location = new Point(24, ClientSize.Height - 53);
            side.Controls.Add(bottom);
            side.Resize += (sender, args) => bottom.Top = side.ClientSize.Height - 53;

            Panel workspace = new Panel { Dock = DockStyle.Fill, BackColor = Ink };
            shell.Controls.Add(workspace, 1, 0);
            formPage.Dock = DockStyle.Fill;
            formPage.AutoScroll = true;
            formPage.BackColor = Ink;
            libraryPage.Dock = DockStyle.Fill;
            libraryPage.BackColor = Ink;
            workspace.Controls.Add(formPage);
            workspace.Controls.Add(libraryPage);
            BuildForm();
            BuildLibrary();
        }

        [System.Runtime.InteropServices.DllImport("user32.dll")]
        private static extern bool DestroyIcon(IntPtr handle);

        private static Icon MakeIcon()
        {
            using (Bitmap bitmap = new Bitmap(64, 64))
            using (Graphics painter = Graphics.FromImage(bitmap)) {
                painter.SmoothingMode = SmoothingMode.AntiAlias;
                using (Brush circle = new SolidBrush(Accent)) painter.FillEllipse(circle, 2, 2, 60, 60);
                using (Pen voice = new Pen(Ink, 4)) {
                    for (int i = 0; i < 4; i++) {
                        int x = 18 + i * 9;
                        int height = i % 2 == 0 ? 24 : 38;
                        painter.DrawLine(voice, x, 32 - height/2, x, 32 + height/2);
                    }
                }
                IntPtr handle = bitmap.GetHicon();
                Icon icon = (Icon)Icon.FromHandle(handle).Clone();
                DestroyIcon(handle);
                return icon;
            }
        }

        private void BuildForm()
        {
            Panel contents = new Panel { Location = new Point(34, 20), Size = new Size(820, 830),
                Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top, BackColor = Ink };
            formPage.Controls.Add(contents);
            formPage.Resize += (sender, args) => {
                contents.Width = Math.Max(560, formPage.ClientSize.Width - 68);
                foreach (Control card in contents.Controls) {
                    if (card is Panel && card.Tag != null && (string)card.Tag == "card") {
                        card.Width = contents.Width;
                    }
                }
            };

            contents.Controls.Add(Label("CRÉER  /  01", 1, 1, 300, 21, 8, Accent, true));
            contents.Controls.Add(Label("La vidéo, dans votre langue.", 0, 26, 780, 52, 23, TextColor, true));
            contents.Controls.Add(Label("Une URL ou un fichier. Une voix suivie d’un bout à l’autre.", 2, 82, 760, 25, 10, Muted, false));

            Panel source = Card(119, 140); source.Tag = "card"; contents.Controls.Add(source);
            source.Controls.Add(Label("01   CHOISIR LA SOURCE", 22, 13, 380, 22, 9, Accent, true));
            source.Controls.Add(Label("Lien YouTube", 22, 43, 180, 19, 9, Muted, false));
            url.Location = new Point(22, 65); url.Size = new Size(644, 32);
            url.AccessibleName = "Lien YouTube";
            url.Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top;
            url.BorderStyle = BorderStyle.FixedSingle; StyleInput(url);
            source.Controls.Add(url);
            url.TextChanged += (sender, args) => {
                if (url.Text.Trim().Length > 0) {
                    selectedFile = null;
                    sourceFileName.Text = "ou choisissez une vidéo sur votre ordinateur";
                }
            };
            Button paste = Button("Coller", Raised, TextColor);
            paste.Location = new Point(680, 63); paste.Size = new Size(110, 34);
            paste.Anchor = AnchorStyles.Right | AnchorStyles.Top;
            paste.Click += (sender, args) => {
                if (Clipboard.ContainsText()) url.Text = Clipboard.GetText().Trim();
            };
            source.Controls.Add(paste);
            Button browseVideo = Button("Choisir un fichier", Raised, TextColor);
            browseVideo.Location = new Point(22, 105); browseVideo.Size = new Size(160, 25);
            browseVideo.Font = new Font("Segoe UI", 8.5f, FontStyle.Regular);
            browseVideo.Click += PickFile;
            source.Controls.Add(browseVideo);
            sourceFileName.Location = new Point(196, 105); sourceFileName.Size = new Size(590, 24);
            sourceFileName.Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top;
            sourceFileName.ForeColor = Muted;
            sourceFileName.Text = "ou choisissez une vidéo sur votre ordinateur";
            source.Controls.Add(sourceFileName);

            Panel options = Card(274, 154); options.Tag = "card"; contents.Controls.Add(options);
            options.Controls.Add(Label("02   PRÉPARER LA VERSION", 22, 13, 430, 22, 9, Accent, true));
            options.Controls.Add(Label("Langue de départ", 22, 44, 230, 20, 9, Muted, false));
            options.Controls.Add(Label("Langue d’arrivée", 300, 44, 230, 20, 9, Muted, false));
            sourceLanguage.Items.AddRange(new object[] { "Anglais", "Détection automatique" });
            targetLanguage.Items.AddRange(new object[] { "Français", "Espagnol", "Allemand", "Italien", "Portugais" });
            sourceLanguage.Location = new Point(22, 67); sourceLanguage.Size = new Size(256, 32);
            targetLanguage.Location = new Point(300, 67); targetLanguage.Size = new Size(256, 32);
            sourceLanguage.AccessibleName = "Langue de départ";
            targetLanguage.AccessibleName = "Langue d’arrivée";
            sourceLanguage.DropDownStyle = targetLanguage.DropDownStyle = ComboBoxStyle.DropDownList;
            StyleInput(sourceLanguage); StyleInput(targetLanguage);
            StyleChoice(sourceLanguage); StyleChoice(targetLanguage);
            options.Controls.Add(sourceLanguage); options.Controls.Add(targetLanguage);
            options.Controls.Add(Label("Voix et traduction", 22, 108, 230, 20, 9, Muted, false));
            mode.Items.AddRange(new object[] {
                "Voix proche · Qwen (un narrateur)",
                "Voix locale · OpenVoice (plusieurs narrateurs)",
                "Piste traduite de YouTube"
            });
            mode.DropDownStyle = ComboBoxStyle.DropDownList;
            mode.Location = new Point(300, 106); mode.Size = new Size(490, 32);
            mode.AccessibleName = "Voix et traduction";
            mode.Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top;
            StyleInput(mode);
            StyleChoice(mode);
            mode.SelectedIndexChanged += (sender, args) => {
                if (mode.SelectedIndex != 2) targetLanguage.SelectedIndex = 0;
            };
            options.Controls.Add(mode);

            Panel destination = Card(443, 103); destination.Tag = "card"; contents.Controls.Add(destination);
            destination.Controls.Add(Label("03   ENREGISTRER LE RÉSULTAT", 22, 13, 430, 22, 9, Accent, true));
            output.Location = new Point(22, 54); output.Size = new Size(644, 33);
            output.AccessibleName = "Dossier de sortie";
            output.Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top;
            output.BorderStyle = BorderStyle.FixedSingle; StyleInput(output);
            destination.Controls.Add(output);
            Button browseOutput = Button("Parcourir", Raised, TextColor);
            browseOutput.Location = new Point(680, 52); browseOutput.Size = new Size(110, 36);
            browseOutput.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            browseOutput.Click += PickOutput;
            destination.Controls.Add(browseOutput);

            Panel action = Card(561, 104); action.Tag = "card"; contents.Controls.Add(action);
            startButton.Text = "Créer la vidéo française";
            startButton.BackColor = Accent; startButton.ForeColor = Ink;
            startButton.FlatStyle = FlatStyle.Flat;
            startButton.FlatAppearance.BorderSize = 0;
            startButton.Font = new Font("Segoe UI", 11, FontStyle.Bold);
            startButton.Location = new Point(22, 20); startButton.Size = new Size(268, 48);
            startButton.Cursor = Cursors.Hand;
            startButton.Click += async (sender, args) => await StartJob();
            action.Controls.Add(startButton);
            cancelButton.Text = "Annuler";
            cancelButton.Location = new Point(304, 29); cancelButton.Size = new Size(86, 30);
            cancelButton.Enabled = false;
            cancelButton.Click += CancelJob;
            action.Controls.Add(cancelButton);
            watchButton.Text = "Lire la vidéo";
            watchButton.Location = new Point(398, 28); watchButton.Size = new Size(118, 32);
            watchButton.Enabled = false;
            watchButton.Click += (sender, args) => {
                if (lastResultPath != null && File.Exists(lastResultPath)) Process.Start(lastResultPath);
            };
            action.Controls.Add(watchButton);
            status.Text = "Prêt à traduire";
            status.ForeColor = Mint;
            status.Location = new Point(532, 26); status.Size = new Size(256, 40);
            status.Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top;
            status.TextAlign = ContentAlignment.MiddleRight;
            action.Controls.Add(status);
            progress.Location = new Point(22, 80); progress.Size = new Size(766, 6);
            progress.Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top;
            progress.Visible = false;
            action.Controls.Add(progress);

            Panel details = Card(680, 117); details.Tag = "card"; contents.Controls.Add(details);
            details.Controls.Add(Label("JOURNAL DU TRAITEMENT", 22, 11, 390, 20, 8.5f, Muted, true));
            log.Location = new Point(22, 34); log.Size = new Size(766, 66);
            log.AccessibleName = "Journal du traitement";
            log.Anchor = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top;
            log.Multiline = true; log.ScrollBars = ScrollBars.Vertical; log.ReadOnly = true;
            log.BorderStyle = BorderStyle.None; StyleInput(log);
            details.Controls.Add(log);
        }

        private static void StyleChoice(ComboBox control)
        {
            control.FlatStyle = FlatStyle.Flat;
            control.DrawMode = DrawMode.OwnerDrawFixed;
            control.ItemHeight = 27;
            control.DrawItem += (sender, args) => {
                if (args.Index < 0) return;
                bool selected = (args.State & DrawItemState.Selected) != 0;
                using (Brush background = new SolidBrush(selected ? Color.FromArgb(65, 54, 117) : Raised)) {
                    args.Graphics.FillRectangle(background, args.Bounds);
                }
                TextRenderer.DrawText(args.Graphics, control.Items[args.Index].ToString(),
                    control.Font, args.Bounds, TextColor, TextFormatFlags.Left | TextFormatFlags.VerticalCenter);
                args.DrawFocusRectangle();
            };
        }

        private void BuildLibrary()
        {
            libraryPage.Controls.Add(Label("MES VERSIONS", 35, 34, 570, 52, 23, TextColor, true));
            libraryPage.Controls.Add(Label("Les vidéos créées restent dans votre dossier de sortie.", 37, 87, 780, 28, 10, Muted, false));
            recent.Location = new Point(37, 137); recent.Size = new Size(800, 444);
            recent.AccessibleName = "Vidéos françaises créées";
            recent.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            recent.View = View.Details; recent.FullRowSelect = true; recent.GridLines = false;
            recent.BackColor = Surface; recent.ForeColor = TextColor; recent.BorderStyle = BorderStyle.None;
            recent.Columns.Add("Vidéo française", 570); recent.Columns.Add("Créée le", 165);
            recent.DoubleClick += (sender, args) => {
                if (recent.SelectedItems.Count != 1) return;
                string path = (string)recent.SelectedItems[0].Tag;
                if (File.Exists(path)) Process.Start(path);
            };
            libraryPage.Controls.Add(recent);
            Button openFolder = Button("Ouvrir le dossier de sortie", Raised, TextColor);
            openFolder.Location = new Point(37, 604); openFolder.Size = new Size(255, 42);
            openFolder.Anchor = AnchorStyles.Left | AnchorStyles.Bottom;
            openFolder.Click += (sender, args) => {
                string folder = output.Text.Trim();
                if (Directory.Exists(folder)) Process.Start(folder);
                else MessageBox.Show(this, "Ce dossier n’existe pas encore.", "LangSwap");
            };
            libraryPage.Controls.Add(openFolder);
        }

        private static void PaintMark(object sender, PaintEventArgs args)
        {
            Graphics g = args.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            using (Brush brush = new SolidBrush(Accent)) g.FillEllipse(brush, 1, 1, 39, 39);
            using (Pen pen = new Pen(Ink, 3)) {
                for (int i = 0; i < 4; i++) {
                    int x = 12 + i * 5;
                    int h = 9 + (i % 2 == 0 ? 9 : 17);
                    g.DrawLine(pen, x, 21 - h/2, x, 21 + h/2);
                }
            }
        }

        private void ShowPage(bool library)
        {
            formPage.Visible = !library;
            libraryPage.Visible = library;
            if (library) libraryPage.BringToFront(); else formPage.BringToFront();
            createTab.BackColor = library ? Color.Transparent : Raised;
            libraryTab.BackColor = library ? Raised : Color.Transparent;
        }

        private void PickFile(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog()) {
                dialog.Title = "Choisir une vidéo";
                dialog.Filter = "Vidéos|*.mp4;*.mkv;*.webm;*.mov;*.avi;*.m4v|Tous les fichiers|*.*";
                if (dialog.ShowDialog(this) == DialogResult.OK) {
                    selectedFile = dialog.FileName;
                    sourceFileName.Text = Path.GetFileName(selectedFile);
                    url.Text = "";
                }
            }
        }

        private void PickOutput(object sender, EventArgs e)
        {
            using (FolderBrowserDialog dialog = new FolderBrowserDialog()) {
                dialog.Description = "Choisir le dossier des vidéos françaises";
                dialog.SelectedPath = Directory.Exists(output.Text) ? output.Text : defaultOutput;
                if (dialog.ShowDialog(this) == DialogResult.OK) output.Text = dialog.SelectedPath;
            }
        }

        private void RefreshRecent()
        {
            recent.Items.Clear();
            if (!Directory.Exists(output.Text)) return;
            FileInfo[] videos = new DirectoryInfo(output.Text).GetFiles("LangSwap-*.mp4");
            Array.Sort(videos, (a, b) => b.LastWriteTime.CompareTo(a.LastWriteTime));
            for (int i = 0; i < Math.Min(100, videos.Length); i++) {
                ListViewItem row = new ListViewItem(videos[i].Name);
                row.SubItems.Add(videos[i].LastWriteTime.ToString("yyyy-MM-dd HH:mm"));
                row.Tag = videos[i].FullName;
                recent.Items.Add(row);
            }
        }

        private static string Quote(string value)
        {
            StringBuilder result = new StringBuilder("\"");
            int slashes = 0;
            foreach (char c in value) {
                if (c == '\\') { slashes++; continue; }
                if (c == '"') {
                    result.Append('\\', slashes * 2 + 1);
                    result.Append('"');
                    slashes = 0;
                    continue;
                }
                result.Append('\\', slashes);
                slashes = 0;
                result.Append(c);
            }
            result.Append('\\', slashes * 2);
            result.Append('"');
            return result.ToString();
        }

        private static bool IsYouTube(string text)
        {
            Uri uri;
            if (!Uri.TryCreate(text, UriKind.Absolute, out uri) || uri.Scheme != Uri.UriSchemeHttps) return false;
            return uri.Host == "youtube.com" || uri.Host == "www.youtube.com"
                || uri.Host == "m.youtube.com" || uri.Host == "youtu.be";
        }

        private async Task StartJob()
        {
            if (busy) return;
            string link = url.Text.Trim();
            string local = selectedFile;
            bool hasLink = link.Length > 0;
            if (hasLink && !IsYouTube(link)) { Fail("Collez une URL YouTube HTTPS valide."); return; }
            if (!hasLink && (local == null || !File.Exists(local))) {
                Fail("Collez une URL YouTube ou choisissez une vidéo."); return;
            }
            bool youtube = mode.SelectedIndex == 2;
            bool qwen = mode.SelectedIndex == 0;
            if (youtube && !hasLink) { Fail("La piste traduite YouTube nécessite une URL."); return; }
            if (qwen && !hasLink) { Fail("La voix Qwen demande une URL YouTube. Pour un fichier, choisissez OpenVoice."); return; }
            if (!youtube && targetLanguage.SelectedIndex != 0) {
                Fail("La voix locale prend actuellement en charge le français."); return;
            }
            string folder = output.Text.Trim();
            if (folder.Length == 0) { Fail("Choisissez un dossier de sortie."); return; }
            if (!Directory.Exists(backendRoot) || !File.Exists(Path.Combine(backendRoot, "Dockerfile.french"))) {
                Fail("Les fichiers de traitement manquent dans ce paquet."); return;
            }
            string target = new[] { "fr", "es", "de", "it", "pt" }[targetLanguage.SelectedIndex];
            string source = sourceLanguage.SelectedIndex == 0 ? "english" : "";
            string jobContainer = "langswap-desktop-" + Guid.NewGuid().ToString("N").Substring(0, 12);
            try {
                Directory.CreateDirectory(folder);
                Directory.CreateDirectory(dataDirectory);
                Directory.CreateDirectory(modelDirectory);
                busy = true;
                startButton.Enabled = false;
                cancelButton.Enabled = true;
                watchButton.Enabled = false;
                progress.Style = ProgressBarStyle.Marquee;
                progress.Visible = true;
                log.Text = "";
                SetStatus("Préparation du moteur local...", Muted);
                if (!imageReady) {
                    string build = "build --progress=plain -f " + Quote(Path.Combine(backendRoot, "Dockerfile.french"))
                        + " -t " + BaseImage + " " + Quote(backendRoot);
                    if (await RunProcess("docker", build) != 0) throw new InvalidOperationException("La préparation du moteur a échoué.");
                    imageReady = true;
                }
                if (qwen && !qwenImageReady) {
                    SetStatus("Préparation du modèle de voix Qwen...", Muted);
                    string buildQwen = "build --progress=plain -f " + Quote(Path.Combine(backendRoot, "Dockerfile.qwen"))
                        + " -t " + QwenImage + " " + Quote(backendRoot);
                    if (await RunProcess("docker", buildQwen) != 0) throw new InvalidOperationException("Le modèle Qwen n’a pas pu être préparé.");
                    qwenImageReady = true;
                }
                string image = qwen ? QwenImage : BaseImage;
                string arguments = "run --rm --name " + jobContainer + (youtube ? "" : " --gpus all")
                    + " -e LANGSWAP_DATA_DIR=/app/data -e LANGSWAP_OUTPUT_DIR=/app/output"
                    + " -v " + Quote(modelDirectory + ":/app/models_weights")
                    + " -v " + Quote(dataDirectory + ":/app/data")
                    + " -v " + Quote(folder + ":/app/output");
                if (!hasLink) arguments += " -v " + Quote(Path.GetDirectoryName(local) + ":/input:ro");
                arguments += " --entrypoint python " + image
                    + " -u desktop_worker.py --mode " + (youtube ? "youtube" : (qwen ? "qwen" : "local"))
                    + (hasLink ? " --url " + Quote(link) : " --file " + Quote("/input/" + Path.GetFileName(local)))
                    + " --source-language " + Quote(source)
                    + " --target-language " + target + " --output-dir /app/output";
                currentContainer = jobContainer;
                SetStatus("Création de la vidéo en cours...", Accent);
                if (await RunProcess("docker", arguments) != 0) throw new InvalidOperationException("Le traitement n’a pas produit de vidéo.");
                SetStatus("Vidéo et sous-titres prêts", Mint);
                RefreshRecent();
            } catch (System.ComponentModel.Win32Exception) {
                Fail("Docker Desktop est introuvable. Installez-le et démarrez-le.");
            } catch (Exception ex) {
                Fail(ex.Message);
            } finally {
                currentContainer = null;
                busy = false;
                startButton.Enabled = true;
                cancelButton.Enabled = false;
                progress.Visible = false;
            }
        }

        private async Task<int> RunProcess(string executable, string arguments)
        {
            return await Task.Run(() => {
                using (Process process = new Process()) {
                    currentProcess = process;
                    process.StartInfo = new ProcessStartInfo(executable, arguments) {
                        UseShellExecute = false, CreateNoWindow = true,
                        RedirectStandardOutput = true, RedirectStandardError = true,
                        StandardOutputEncoding = Encoding.UTF8,
                        StandardErrorEncoding = Encoding.UTF8,
                    };
                    process.OutputDataReceived += (sender, data) => HandleLine(data.Data);
                    process.ErrorDataReceived += (sender, data) => HandleLine(data.Data);
                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();
                    process.WaitForExit();
                    currentProcess = null;
                    return process.ExitCode;
                }
            });
        }

        private void HandleLine(string value)
        {
            if (value == null || IsDisposed || !IsHandleCreated) return;
            BeginInvoke((MethodInvoker)(() => {
                if (value.StartsWith("LANGSWAP_STAGE|")) {
                    SetStatus(value.Substring("LANGSWAP_STAGE|".Length), Accent);
                } else if (value.StartsWith("LANGSWAP_ERROR|")) {
                    Fail(value.Substring("LANGSWAP_ERROR|".Length));
                } else if (value.StartsWith("LANGSWAP_RESULT|")) {
                    string[] parts = value.Split('|');
                    if (parts.Length > 1) {
                        lastResultPath = Path.Combine(output.Text.Trim(), parts[1]);
                        watchButton.Enabled = File.Exists(lastResultPath);
                        AppendLine("Vidéo créée : " + parts[1]);
                    }
                } else if (value.StartsWith("#") &&
                           (value.IndexOf("Downloading", StringComparison.OrdinalIgnoreCase) >= 0
                            || value.IndexOf("DONE", StringComparison.OrdinalIgnoreCase) >= 0
                            || value.IndexOf("exporting", StringComparison.OrdinalIgnoreCase) >= 0)) {
                    AppendLine(value);
                } else if (value.IndexOf("error", StringComparison.OrdinalIgnoreCase) >= 0) {
                    AppendLine(value);
                }
            }));
        }

        private void AppendLine(string value)
        {
            log.AppendText(value + Environment.NewLine);
        }

        private void SetStatus(string value, Color color)
        {
            status.Text = value;
            status.ForeColor = color;
            AppendLine(value);
        }

        private void Fail(string reason)
        {
            status.Text = reason;
            status.ForeColor = Color.FromArgb(255, 145, 145);
            AppendLine(reason);
        }

        private void CancelJob(object sender, EventArgs e)
        {
            if (currentContainer != null) {
                string id = currentContainer;
                Task.Run(() => {
                    using (Process stop = Process.Start(new ProcessStartInfo("docker", "stop " + id) {
                        UseShellExecute = false, CreateNoWindow = true })) {
                        if (stop != null) stop.WaitForExit(15000);
                    }
                });
            }
            Process process = currentProcess;
            if (process != null && !process.HasExited) process.Kill();
            SetStatus("Traitement annulé. Les données partielles restent privées.", Muted);
        }
    }

}
