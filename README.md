# Classification Laboratory — Karar Sınırı ve Sınıflandırma Laboratuvarı

**Classification Laboratory**, sınıflandırma algoritmalarının iki boyutlu veri üzerindeki davranışlarını incelemek amacıyla geliştirilmiş bir **C# / .NET 8 Windows Forms** masaüstü uygulamasıdır. Kullanıcı farklı geometrilerde veri kümeleri oluşturabilir, noktaları fareyle düzenleyebilir ve **Binary Perceptron**, **Multiclass Perceptron** ile **Multi-Layer Perceptron (MLP)** modellerinin karar bölgelerini görsel olarak karşılaştırabilir.

> **Temel hedef:** Lineer karar sınırlarının hangi problemlerde yeterli olduğunu, XOR ve spiral gibi örneklerde doğrusal olmayan MLP'nin nasıl farklı davrandığını; eğitim, görselleştirme ve test metrikleriyle göstermek.

**Teknolojiler:** C# · .NET 8 · Windows Forms · elle yazılmış ML algoritmaları · GitHub Actions  
**İşletim sistemi:** Windows · **Çözüm:** `ClassificationLaboratory.sln` · **Başlangıç projesi:** `ClassificationApp`

## Temsili arayüz önizlemesi

> **Önemli:** Bu bölümde kullanılacak görsel, projenin kaynak kodundaki arayüz düzeni ve tema renkleri esas alınarak **yapay zekâ ile oluşturulmuş temsili bir tasarımdır**. **Gerçek uygulama ekran görüntüsü değildir.** Görseldeki grafikler, metrikler, eğitim sonuçları, tahmin güvenleri ve süreler **örnek değerlerdir; çalıştırılıp ölçülmüş sonuçlar olarak yorumlanmamalıdır.**

<!-- AI_ARAYUZ_GORSEL_BASLANGIC -->
*Görsel GitHub deposunun `assets/screenshots/ai-temsili-arayuz.png` yoluna eklendiğinde burada gösterilecektir.*
<!-- AI_ARAYUZ_GORSEL_BITIS -->

## İçindekiler

1. [Başlıca özellikler](#başlıca-özellikler)
2. [Kurulum ve çalıştırma](#kurulum-ve-çalıştırma)
3. [İlk deney: doğrusal ve doğrusal olmayan sınırlar](#ilk-deney-doğrusal-ve-doğrusal-olmayan-sınırlar)
4. [Veri kümeleri ve grafik etkileşimleri](#veri-kümeleri-ve-grafik-etkileşimleri)
5. [Algoritmalar ve eğitim](#algoritmalar-ve-eğitim)
6. [Ölçümler ve analiz panelleri](#ölçümler-ve-analiz-panelleri)
7. [Proje mimarisi](#proje-mimarisi)
8. [Sınırlamalar, CI ve sorun giderme](#sınırlamalar-ci-ve-sorun-giderme)

## Başlıca özellikler

| Alan | Uygulamadaki karşılığı |
| --- | --- |
| Hazır veri | **Linearly Separable**, **XOR**, **Clusters** ve **Spiral** |
| Sınıf yönetimi | 2–5 sınıf; aktif sınıf, renk ve işaretçi (marker) seçimi |
| Etkileşim | Sol tıkla nokta ekleme/seçme, sürükleme, sağ tıkla silme |
| Veri bölme | Sınıf oranlarını gözeten **stratified train/test split** |
| Algoritmalar | Binary Perceptron, Multiclass Perceptron, MLP |
| Karar analizi | Karar bölgeleri, sınır çizgileri, epoch anlık görüntülerinden animasyon |
| Görsel filtreler | Train/test görünümü, yanlış sınıflandırılan örnekleri vurgulama |
| Metrikler | Accuracy, Macro Precision, Macro Recall, Macro F1 |
| Tanı araçları | Confusion matrix, loss/hata geçmişi, eğitim logu |
| Karşılaştırma | Aynı veri kümesinde model çalıştırmalarının listesi ve eski modelin tekrar gösterimi |

## Kurulum ve çalıştırma

### Gereksinimler

- **Windows 10/11**.
- **.NET 8 SDK**.
- IDE ile geliştirme için **Visual Studio 2022** ve **.NET desktop development** bileşeni.
- Makine öğrenmesi kısmı için TensorFlow, PyTorch, ML.NET veya Accord.NET gibi hazır eğitim kütüphanelerine ihtiyaç yoktur.

### Depoyu indirme ve derleme

~~~powershell
git clone https://github.com/seydivakkas/ClassificationLaboratory_ModernDashboard.git
cd ClassificationLaboratory_ModernDashboard
dotnet restore ClassificationLaboratory.sln
dotnet build ClassificationLaboratory.sln --configuration Release --no-restore
~~~

### Uygulamayı başlatma

~~~powershell
dotnet run --project ClassificationApp/ClassificationApp.csproj --configuration Release
~~~

**Visual Studio ile:** `ClassificationLaboratory.sln` dosyasını açın, `ClassificationApp` projesini başlangıç projesi seçin ve **F5** tuşuna basın.

> `ML.Core`, `net8.0` hedefli algoritma kütüphanesidir; arayüz `ClassificationApp` altında `net8.0-windows` hedeflidir. Grafik uygulaması Windows'a yöneliktir.

## İlk deney: doğrusal ve doğrusal olmayan sınırlar

### Deney A — XOR

1. Soldaki **Dataset** bölümünden **XOR**'u seçin. XOR iki sınıflı olduğu için uygulama sınıf sayısını **2** olarak ayarlar.
2. **Model & Training** alanından **Binary** algoritmasını seçin.
3. **MODELİ EĞİT** düğmesine basın; karar bölgelerini ve metrikleri inceleyin.
4. Aynı veri üzerinde **MLP**'yi seçip tekrar eğitim çalıştırın. Başlangıç için **12 hidden**, learning rate **0.001** ve momentum **0.90** kullanılabilir; hiperparametreler veri setine göre ayarlanmalıdır.
5. **COMPARE** sekmesinde sonuçların **Accuracy** ve **Macro F1** değerlerini karşılaştırın. İsterseniz **Seçili modeli göster** ile önceki bir çalıştırmanın sınırını yeniden görüntüleyin.

**Neden önemli?** XOR, tek bir doğrusal karar sınırıyla tam ayrılamayan tipik bir örnektir. Bu senaryo MLP'nin doğrusal olmayan karar mekanizmasını öğretici şekilde gösterebilir.

### Deney B — Spiral

1. **Spiral** veri kümesini oluşturun; ihtiyaca göre sınıf sayısını belirleyin.
2. **Multiclass** modelini eğitin ve karar bölgelerini gözlemleyin.
3. Aynı veri üzerinde **MLP** eğiterek sınır şekli ve test metriklerini karşılaştırın.
4. **Epoch sınır animasyonu** seçeneğini açık tutarak modelin eğitim sırasındaki değişimini izleyin.

**Önemli:** Örnek sonuçları sabit başarı garantisi değildir. Noise, örnek sayısı, seed, train/test dağılımı ve model hiperparametreleri sonucu değiştirir.

## Veri kümeleri ve grafik etkileşimleri

### Kullanılabilen veri türleri

| Veri türü | İncelenen problem |
| --- | --- |
| **Linearly Separable** | Doğrusal olarak ayrılabilir sınıflar |
| **XOR** | Doğrusal ayrılamayan iki sınıflı ilişki |
| **Clusters** | Kümelerden oluşan örnek dağılımı |
| **Spiral** | Karmaşık ve eğrisel karar bölgeleri |

Kontroller:

- **Örnek / sınıf:** Her sınıf için üretilecek örnek sayısı (**4–250**).
- **Noise:** Rastgele dağılım/gürültü düzeyi (**0.05–2.50**).
- **Random seed:** Tekrarlanabilir sentetik örnek üretimine yardımcı olan başlangıç değeri.
- **Sınıf sayısı:** **2–5**; **Binary** seçeneği iki sınıflı görev içindir.
- **Test oranı (%):** **%10–%50**, varsayılan **%25**.

### Fareyle veri düzenleme

- **Sol tık:** Seçili sınıfa ait yeni bir nokta ekler veya yakındaki noktayı seçer.
- **Sol tuşla sürükleme:** Noktanın konumunu değiştirir.
- **Sağ tık:** Yakındaki noktayı kaldırır.
- **Aktif sınıf:** Soldaki `Class 0`, `Class 1` vb. radyo düğmeleriyle belirlenir.
- **Renk/marker:** Sınıf başına ayrı renk ve **Circle, Square, Triangle, Diamond, Cross** işaretçileri seçilebilir.

Veri değişikliği, önceden eğitilmiş modelin eski veriyle ilişkisinin değiştiği anlamına gelir. Uygulama modelleri veri kümesi sürümüyle ilişkilendirir ve **farklı veri sürümüne ait modelin karar sınırının güvenli karşılaştırılmaması** için kontrol uygular.

### Train/test ayrımı

`MainForm.ApplyStratifiedSplit`, her sınıfı ayrı karıştırıp belirlenen orana göre test örnekleri ayırır. Böylece özellikle az örnekli sınıfların eğitim kümesinden tamamen kaybolması riski azaltılır. Teste ayrılacak yeterli örneği olmayan sınıflarda ilgili örnekler eğitim kümesinde kalabilir; test kümesi boşsa değerlendirme eğitim verisi üzerinden yapılabilir. Bu durumda görülen ölçümü bağımsız test sonucu gibi yorumlamayın.

## Algoritmalar ve eğitim

### Binary Perceptron

- İki sınıflı problemlere uygundur.
- Giriş vektörü **(x, y)** üzerinden doğrusal ayırma öğrenir.
- Hatalı sınıflandırılan örnekler üzerinden ağırlık/güncelleme işlemi uygular.
- Momentum destekler; bir epoch'taki yanlış sayısı eğitim geçmişinde izlenebilir.
- Doğrusal olmayan XOR örneklerini kusursuz çözmesi beklenmemelidir.

**Kod:** `ML.Core/Classification/BinaryPerceptron.cs`

### Multiclass Perceptron

- İkiden fazla sınıf için genel perceptron yaklaşımını uygular.
- Her sınıfa karşılık gelen skor/ağırlık değerleri üzerinden sınıf seçilir.
- Yanlış sınıfın ve doğru sınıfın parametreleri güncellenir.
- Karar sınırları doğrusal model ailesinin kapasitesiyle sınırlıdır.

**Kod:** `ML.Core/Classification/MulticlassPerceptron.cs`

### Multi-Layer Perceptron (MLP)

Ağ düzeni: **2 giriş → ayarlanabilir gizli katman → sınıf sayısı kadar çıkış**.

- Gizli katmanda **sigmoid** aktivasyonu.
- Çıkışta **softmax** ile sınıf skorlarının normalize edilmesi.
- Eğitimde **cross-entropy** kaybı ve elle yazılmış geri yayılım.
- Güncellemede learning rate ve momentum.
- Gizli nöron sayısı **2–128** arasında ayarlanabilir.

**Kod:** `ML.Core/Classification/MlpClassifier.cs`

### Ortak eğitim altyapısı

- `ZScoreScaler`: Özelliklerin eğitim verisine göre standardizasyonu.
- `IClassifier`: `Fit`, `Predict` ve `PredictScores` sözleşmesi.
- `IProgressiveClassifier`: `FitWithProgress` üzerinden epoch anlık görüntüleri alma.
- `ClassifierEpochSnapshot` ve `FrozenClassifier`: Eğitim sürecindeki karar yüzeylerini tekrar çizebilmek için model anlık durumlarının taşınması.
- Eğitim parametreleri: learning rate (**varsayılan 0.001**), max epoch (**10000**), momentum (**0.90**) ve MLP hidden (**12**).
- Arayüz, eğitim sırasında snapshot toplayıp **karar sınırı animasyonu** gösterebilir.

## Ölçümler ve analiz panelleri

| Sekme | Gösterilen veri |
| --- | --- |
| **LOSS** | Epoch boyunca sınıflandırma hatası/kayıp geçmişi |
| **MATRIX** | Gerçek sınıf / tahmin sınıfı dağılımını gösteren karmaşıklık matrisi |
| **COMPARE** | Model, Accuracy, F1, epoch ve loss karşılaştırma listesi |
| **LOG** | Eğitim başlangıcı, bitişi, parametreler, hata ve durum kayıtları |

Başlıca değerlendirme metrikleri:

- **Accuracy:** Doğru tahmin sayısının değerlendirme örneklerine oranı.
- **Precision:** Bir sınıfa yapılan pozitif tahminlerin ne kadarının doğru olduğu.
- **Recall:** Gerçek sınıf örneklerinin ne kadarının bulunduğu.
- **Macro F1:** Sınıf bazlı F1 değerlerinin aritmetik ortalaması; sınıfları destek sayısından bağımsız olarak ele alır.

Arayüzdeki **Precision / Recall** kartları sınıf bazlı değil, **macro** özetlerini kullanır. Karar bölgesi ve sınır çizgisi, yanlış sınıflandırma vurgusu ve train/test işaretleme seçenekleri bağımsız olarak açılıp kapatılabilir.

**Model geçmişi:** Aynı veri setinde birden çok çalışma `COMPARE` tablosunda saklanır. Bu geçmiş uygulama belleğindedir; **kalıcı model dosyası kaydetme/yükleme özelliği bulunmaz**.

## Proje mimarisi

~~~text
ClassificationLaboratory_ModernDashboard/
├── ClassificationLaboratory.sln
├── ClassificationApp/                  # Windows Forms sunum katmanı
│   ├── MainForm.cs                      # Veri, eğitim, metrik ve olay akışı
│   ├── DatasetGenerator.cs              # Linearly Separable / XOR / Clusters / Spiral
│   ├── PlotPanel.cs                     # Etkileşimli noktalar ve karar bölgeleri
│   ├── ConfusionMatrixPanel.cs
│   ├── LossChartPanel.cs
│   ├── ModelRun.cs                       # Eğitim karşılaştırma kaydı
│   ├── UiTheme.cs
│   └── ClassificationApp.csproj
├── ML.Core/
│   ├── Classification/
│   │   ├── IClassifier.cs
│   │   ├── IProgressiveClassifier.cs
│   │   ├── BinaryPerceptron.cs
│   │   ├── MulticlassPerceptron.cs
│   │   ├── MlpClassifier.cs
│   │   ├── ClassifierEpochSnapshot.cs
│   │   └── ClassificationEvaluation.cs
│   ├── Preprocessing/ZScoreScaler.cs
│   └── ML.Core.csproj
└── .github/workflows/build.yml
~~~

**Veri akışı:** Sentetik veri/fare düzenlemesi → stratified split → ölçekleme → model eğitimi → tahmin ve karar bölgeleri → confusion matrix/metrikler → karşılaştırma.

## Sınırlamalar, CI ve sorun giderme

### CI ve doğrulama

`.github/workflows/build.yml`, **push** ve **pull request** olaylarında Windows runner üzerinde .NET 8 ile **restore + Release build** çalıştırır. Sonuçlar [GitHub Actions](https://github.com/seydivakkas/ClassificationLaboratory_ModernDashboard/actions) ekranındadır.

**Bu depoda henüz otomatik unit/integration test projesi yoktur.** Derlemenin başarılı olması sınıflandırma sonuçlarının matematiksel olarak doğrulandığı veya tüm UI senaryolarının test edildiği anlamına gelmez.

### Bilinen kapsam sınırları

- Görsel arayüz **Windows Forms** tabanlıdır ve Windows hedeflenir.
- Veri kümeleri uygulamanın oluşturduğu sentetik verilerden veya fareyle eklenen noktalardan oluşur; **CSV import/export bulunmaz**.
- Model geçmişi **uygulama oturumu süresince** tutulur.
- Kapsam geniş, çok katmanlı genel amaçlı bir derin öğrenme çatısı değildir; model mimarileri eğitim amaçlı sabittir.
- Otomatik test/benchmark ve doğrulanmış başarı yüzdeleri bu dokümana eklenmemiştir.

### Sık karşılaşılan durumlar

**Eğitim başlamıyor:** En az **4** örnek bulunmalı ve seçili sınıf aralığındaki **her sınıf en az bir örnekle** temsil edilmelidir.  
**Binary modeli hata veriyor:** İki sınıflı bir veri kümesi seçin.  
**Eski model gösterilmiyor:** Veri kümesi değişmiş olabilir; aynı veri sürümünde yeniden eğitim yapın.  
**XOR üzerinde doğrusal model başarısız:** Bu durum modelin doğrusal sınır kapasitesiyle ilişkili olabilir; MLP ile karşılaştırın.  
**`dotnet` veya masaüstü bileşeni eksik:** .NET 8 SDK ve Visual Studio masaüstü iş yükünü kontrol edin.

### Lisans ve kullanım amacı

Depoda `LICENSE` dosyası yoktur. **Public repo olması, otomatik açık kaynak lisansı anlamına gelmez.** Kullanım/dağıtım koşullarını depo sahibi belirlemelidir.

Bu uygulama **öğretim ve deneysel görselleştirme** içindir. Gerçek dünya sınıflandırma doğruluğu veya üretim kullanımına ilişkin bir garanti verilmez.
