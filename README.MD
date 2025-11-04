# Sıfırdan LLM — Kurs Özeti ve Bu Depo

Bu depo, "sıfırdan LLM tasarlama" kursunda anlatılan teorinin ve pratik adımların kısa bir özetini içerir. Notebook ve yardımcı dosyalarla küçük bir deney ortamı sağlanmıştır. Aşağıda kursun ana fikirleri, bu repodaki dosyaların açıklaması ve çalışma / hata giderme notları (özellikle "CPU dispatcher tracer already initialized" hatası için) bulunmaktadır.

## Hızlı Bakış

- Amaç: Küçük bir dil modeli (tokenization -> model -> eğitim -> çıkarım) tasarımının teorisini ve pratik adımlarını anlamak.
- Temel bileşenler: Tokenizasyon, sözlük/vocab, altkelime (BPE/SentencePiece), Transformer mimarisi (self-attention), eğitim döngüsü (loss, optimizer, batching), değerlendirme ve deploy.

## Ana Teorik Konular 

1. Tokenizasyon
   - Kelime tabanlı tokenizasyon yerine altkelime (subword) yöntemleri (BPE, SentencePiece) daha az bilinmeyen token ve daha iyi genel performans sağlar.
   - Tokenizer, girdi metni sayısal idlere çevirir; model bu id dizisini girdi olarak alır.

2. Transformer ve Self-Attention
   - Transformer: katmanlar halinde multi-head self-attention + feed-forward blokları.
   - Self-attention, her tokenin diğer tokenlerle bağlam ilişkisini öğrenmesini sağlar.
   - Pozisyonel kodlamalar dizindeki sıralamayı modele sağlar.

3. Eğitim Pratiği
   - Masked / causal language modeling: eğitim hedefi sıralı tahmin (ör. causal LM) veya maskelenmiş tahmin olabilir.
   - Batch boyutu, sequence length, learning rate schedule, weight decay, optimizer (AdamW) kritik hyperparametrelerdir.
   - Gradient accumulation, mixed-precision (fp16/bf16), ve dağıtık eğitim büyük modeller için önemlidir.

4. Değerlendirme ve Oluşturma
   - Perplexity, loss, n-gram hataları gibi metriklerle değerlendirme.
   - Sampling (top-k, top-p), beam search ile oluşturma stratejileri.

5. Güvenlik & Etik
   - Veri kalitesi, telif, zararlı içerik filtreleme ve modelin yanlış kullanım risklerine dikkat.


## Nasıl Çalıştırılır (Hızlı)

Önerilen Python paketlerini kurun (örnek):

```powershell
python -m pip install --upgrade pip
pip install sentencepiece tokenizers transformers tiktoken
```

