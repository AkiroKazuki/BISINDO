# BAB I PENDAHULUAN

## A.1 Latar Belakang

Disabilitas merupakan isu sosial yang masih menjadi permasalahan dunia (Adioetomo et al., 2014; Andayani, 2015; Boland, 1992; Plegè et al., 2015; Murugami, 2009; Santoso & Apsari, 2017). Penyandang disabilitas menurut Undang-Undang Nomor 8 Tahun 2016 tentang Penyandang Disabilitas (UUPD) adalah setiap orang yang memiliki keterbatasan fisik, intelektual, mental, dan sensorik dalam jangka waktu lama yang dalam berinteraksi dengan lingkungan dapat mengalami hambatan dan kesulitan untuk berpartisipasi secara penuh dan efektif dengan warga negara lainnya berdasarkan kesamaan hak.

Berdasarkan data Badan Pusat Statistik (BPS) tahun 2023, jumlah penyandang disabilitas pendengaran di Indonesia mencapai sekitar 22,97 juta jiwa. Sementara WHO dan ILO mencatat jumlah penyandang disabilitas secara keseluruhan di Indonesia mencapai 41 juta orang dari 275 juta penduduk pada tahun 2022 (Kordi, 2023). Data ini menunjukkan bahwa populasi penyandang tuli di Indonesia cukup signifikan dan memerlukan perhatian khusus dalam hal aksesibilitas komunikasi.

Salah satu alat bantu komunikasi yang banyak digunakan oleh penyandang tuli adalah bahasa isyarat (*sign language*). Sistem bahasa isyarat yang populer di Indonesia terbagi menjadi dua, yaitu SIBI (Sistem Isyarat Bahasa Indonesia) yang diturunkan dari *American Sign Language* (ASL), dan BISINDO (Bahasa Isyarat Indonesia) yang merupakan bahasa isyarat alami komunitas tuli Indonesia. BISINDO lebih sering digunakan dalam komunikasi sehari-hari karena sifatnya yang lebih spontan dan natural dibanding SIBI yang bersifat baku.

Permasalahan utama yang dihadapi penyandang tuli bukan hanya kurangnya fasilitas, melainkan minimnya pemahaman masyarakat awam terhadap bahasa isyarat. Akibatnya, sering terjadi salah paham atau komunikasi yang terputus, baik di sekolah, di tempat kerja, maupun di lingkungan sekitar (Santoso & Apsari, 2017). Hambatan lingkungan sosial menyebabkan penyandang disabilitas mengalami kesulitan partisipasi penuh di masyarakat (Malik et al., 2024). **Kondisi ini menjadi sangat kritis dalam situasi darurat**, di mana penyandang tuli kesulitan untuk menyampaikan isyarat darurat seperti "TOLONG", "BAHAYA", atau "KEBAKARAN" kepada orang-orang di sekitarnya yang tidak memahami bahasa isyarat. Oleh karena itu, diperlukan sebuah sistem yang dapat berfungsi sebagai **jembatan komunikasi** (*communication bridge*) yang menghubungkan penyandang disabilitas pendengaran/wicara dengan masyarakat umum yang tidak memahami bahasa isyarat, khususnya dalam situasi krusial ini.

Lebih lanjut, sistem peringatan dini bencana (*Early Warning System*/EWS) yang ada di Indonesia sebagian besar masih berbasis suara seperti sirine dan pengumuman *speaker*, sehingga tidak dapat diakses oleh penyandang tuli. Penelitian menunjukkan bahwa penyandang tuli hampir tidak dapat mengenali isyarat peringatan bahaya dari EWS berbasis suara (Unihaz, 2024). Dalam konteks Indonesia sebagai negara rawan bencana dengan rata-rata 5.000+ gempa per tahun dan banjir rutin di berbagai daerah, kebutuhan akan sistem komunikasi darurat yang inklusif menjadi sangat mendesak.

Seiring kemajuan teknologi, berbagai solusi dikembangkan untuk mengatasi hambatan komunikasi tersebut. Kombinasi kamera dan *deep learning* memungkinkan pengembangan sistem pengenalan bahasa isyarat yang menerjemahkan gerakan tangan ke teks atau suara, sehingga mudah dipahami oleh masyarakat awam (Candra & Kusrini, 2025). Namun, penerapan teknologi pengenalan isyarat masih menghadapi beberapa kendala. Lu et al. (2023) menjelaskan bahwa kondisi lingkungan sangat mempengaruhi kinerja sistem. Pencahayaan yang terlalu terang atau terlalu gelap dapat mempengaruhi kamera dalam menangkap bentuk tangan secara jelas. Selain itu, apabila sebagian tangan atau jari tertutup (oklusi parsial), maka informasi gerakan tidak dapat tertangkap secara utuh sehingga isyarat menjadi salah dikenali. Oleh karena itu, dibutuhkan pendekatan berbasis kerangka (*skeleton-based*) dalam mengenali bahasa isyarat tersebut.

Model CNN (*Convolutional Neural Network*) dan RNN (*Recurrent Neural Network*) memang sering digunakan dalam penelitian pengenalan bahasa isyarat, namun keduanya masih memiliki kekurangan. CNN dapat mengenali pola visual pada satu gambar, seperti bentuk tangan atau posisi jari. Akan tetapi ketika pencahayaan berubah, tangan sebagian tertutup, atau pengguna memiliki gaya gerakan yang berbeda, akurasi sistem dapat menurun secara signifikan (Rakhmadi et al., 2025). Jika hanya mengandalkan CNN, sistem sering mengalami kesulitan membedakan gerakan yang tampak mirip tetapi memiliki urutan berbeda, karena CNN tidak dirancang untuk memahami alur gerakan secara berkelanjutan (Ugale et al., 2023).

Sedangkan RNN dirancang untuk memproses data secara berurutan, sehingga sering dikombinasikan dengan CNN untuk menangani gerakan dinamis. Namun, RNN membutuhkan sumber daya komputasi yang besar dan waktu pelatihan yang relatif lama. Selain itu, RNN sering mengalami masalah *vanishing gradient* yang menyebabkan kesulitan dalam mengingat urutan yang panjang. Pada bahasa isyarat berbentuk kalimat dengan banyak gerakan, RNN dapat kehilangan informasi pada bagian awal dan lebih fokus pada bagian akhir, sehingga hasil terjemahan menjadi kurang akurat.

Untuk mengatasi keterbatasan tersebut, *Spatial-Temporal Graph Convolutional Network* (ST-GCN) hadir sebagai solusi yang lebih optimal. ST-GCN adalah model *deep learning* yang dirancang untuk memproses data berbentuk graf yang berubah seiring waktu, dengan tetap memperhatikan interaksi antar *node* dalam sebuah grafik (Zhao & Chen, 2023). Kelebihan yang paling menonjol dari ST-GCN adalah kemampuannya untuk sekaligus menangkap hubungan spasial antar sendi tubuh (seperti jari, pergelangan, siku, dan bahu) dan dinamika perubahan gerakan seiring waktu dalam satu kerangka kerja yang terintegrasi. ST-GCN melakukan penggabungan antara operasi konvolusi pada struktur grafik dengan konvolusi temporal, sehingga model ini dapat mengenali pola keterkaitan antar *node* dan perubahannya seiring waktu secara bersamaan.

Selain itu, ST-GCN cenderung lebih efisien secara komputasi dibandingkan dengan metode yang menggabungkan *graph* dengan jaringan berulang seperti RNN atau LSTM. Arsitektur ini menerapkan konvolusi untuk mengolah urutan waktu, sehingga tidak memerlukan langkah pemrosesan sekuensial panjang yang membuat pelatihan menjadi lambat atau rentan terhadap masalah gradien (Zhao & Chen, 2023). Model ST-GCN juga telah terbukti efektif dalam berbagai aplikasi pengenalan aksi berbasis *skeleton*, termasuk deteksi perilaku anomali (Song et al., 2025).

Dalam penelitian ini, ST-GCN dikombinasikan dengan MediaPipe Holistic untuk ekstraksi *skeleton* (kerangka tubuh) secara *real-time*. MediaPipe menghasilkan 75 titik *landmark* (33 pose tubuh + 21 tangan kiri + 21 tangan kanan) yang kemudian dimodelkan sebagai graf untuk diproses oleh ST-GCN. Pendekatan berbasis *skeleton* ini memiliki keunggulan dalam mengatasi variasi pencahayaan karena tidak bergantung pada warna atau tekstur kulit, serta lebih *robust* terhadap oklusi parsial karena hubungan antar sendi dapat diinferensikan meskipun sebagian titik tidak terdeteksi.

Sistem yang dikembangkan tidak hanya mendeteksi isyarat, tetapi juga dilengkapi dengan **fitur notifikasi otomatis** berupa *text-to-speech* (TTS) untuk mengucapkan hasil deteksi dan notifikasi SMS/*push* untuk menghubungi kontak darurat. Sistem dibangun sebagai aplikasi berbasis web yang dapat diakses dari perangkat *mobile*, sehingga praktis digunakan dalam berbagai situasi. Dataset yang digunakan mencakup 10 isyarat darurat (TOLONG, BAHAYA, KEBAKARAN, SAKIT, GEMPA, BANJIR, PENCURI, PINGSAN, KECELAKAAN, DARURAT) yang diperkaya dengan *pre-training* dari dataset SIBI alfabet untuk meningkatkan kemampuan generalisasi model.

Berdasarkan uraian di atas, penulis tertarik untuk melakukan penelitian dengan judul **"Pengembangan Sistem Deteksi Isyarat Darurat BISINDO Berbasis Arsitektur ST-GCN dengan Ketahanan Terhadap Variasi Pencahayaan dan Oklusi Parsial"**. Penelitian ini diharapkan dapat membantu penyandang tuli untuk berkomunikasi dalam situasi darurat secara *real-time* dengan masyarakat umum, serta berkontribusi dalam menciptakan lingkungan yang lebih inklusif dan aman bagi semua warga negara Indonesia.

---

## A.2 Tujuan Penelitian

1. Mengembangkan sistem deteksi isyarat darurat BISINDO berbasis arsitektur ST-GCN yang dapat mengenali 10 jenis isyarat darurat secara *real-time*.
2. Menguji ketahanan (*robustness*) sistem terhadap variasi kondisi pencahayaan (terang, normal, redup, gelap).
3. Menguji ketahanan sistem terhadap oklusi parsial pada berbagai tingkat (0%, 25%, 50%).
4. Mengintegrasikan sistem dengan fitur notifikasi otomatis (*text-to-speech* dan SMS) untuk mendukung komunikasi darurat.

---

## A.3 Manfaat Penelitian

### Manfaat Praktis
1. Membantu penyandang tuli untuk menyampaikan isyarat darurat kepada masyarakat umum secara *real-time* tanpa bergantung pada juru bahasa isyarat.
2. Mempercepat respons pertolongan dalam situasi darurat melalui fitur notifikasi otomatis ke kontak darurat.
3. Meningkatkan kemandirian dan rasa aman penyandang tuli dalam beraktivitas di ruang publik.

### Manfaat Teoritis
1. Memberikan kontribusi dalam pengembangan teknologi aksesibilitas bagi penyandang disabilitas di Indonesia.
2. Menyediakan *benchmark* performa ST-GCN untuk pengenalan bahasa isyarat BISINDO dalam berbagai kondisi lingkungan.
3. Memperkaya literatur penelitian mengenai sistem deteksi isyarat yang fokus pada isyarat darurat, bukan hanya alfabet atau kata umum.

---

## DAFTAR PUSTAKA

Adioetomo, S. M., Mont, D., & Irwanto. (2014). *Persons with Disabilities in Indonesia: Empirical Facts and Implications for Social Protection Policies*. TNP2K.

Andayani, R. D. (2015). Pemenuhan Hak Penyandang Disabilitas dalam Kebijakan Publik. *Proceedings Seminar Nasional Sosiologi*, 1(1), 1–12.

Badan Pusat Statistik. (2023). *Statistik Kesejahteraan Rakyat 2023*. BPS RI.

Boland, J. A. (1992). Disability and Social Policy: An Evaluation of Practice. *Journal of Disability Policy Studies*, 3(1), 65–80.

Candra, K. K., & Kusrini, K. (2025). Klasifikasi Gambar Bahasa Isyarat Indonesia (Bisindo) Pada Komunitas Tuli Menggunakan Machine Learning. *E-Jurnal JUSITI*, 14(1), 56–63. https://doi.org/10.36774/jusiti.v14i1.1649

Kordi, M. G. H. (2023). Melibatkan Disabilitas. *BaKTINews - Yayasan BaKTI*. https://baktinews.bakti.or.id/artikel/melibatkan-disabilitas

Lu, C., Kozakai, M., & Jing, L. (2023). Sign Language Recognition with Multimodal Sensors and Deep Learning Methods. *Electronics*, 12(23), 4827. https://doi.org/10.3390/electronics12234827

Malik, D. A., Ningrum, N. A., Zulvia, R. A., Alfani, F. S., Aisah, P. M. R., & Saebani, B. A. (2024). Perlindungan Hak Penyandang Disabilitas dalam Perspektif Sosiologi Hukum. *Legal Standing: Jurnal Ilmu Hukum*, 8(3a), 871–881. https://doi.org/10.24269/ls.v8i3a.10718

Murugami, M. W. (2009). Disability and Identity. *Disability Studies Quarterly*, 29(4). https://doi.org/10.18061/dsq.v29i4.988

Plegè, K., Bergkamp, A., & Meijer, W. (2015). Disability in the Global South: The Critical Handbook. *International Review of Social Sciences*, 4(2), 115–130.

Rakhmadi, A., Yudhana, A., & Sunardi, S. (2025). A Study of Worldwide Patterns in Alphabet Sign Language Recognition Using Convolutional and Recurrent Neural Networks. *Jurnal Teknik Informatika (JUTIF)*, 6(1), 187–204. https://doi.org/10.20884/1.jutif.2025.6.1.4202

Santoso, M. B., & Apsari, N. C. (2017). Pergeseran Paradigma dalam Disabilitas. *Intermestic: Journal of International Studies*, 1(2), 166–176. https://doi.org/10.24198/intermestic.v1n2.6

Song, G., Qian, Y., & Wang, Y. (2025). STGCN-PAD: A Spatial-Temporal Graph Convolutional Network for Pedestrian Behavior Anomaly Detection at Grade Crossings. *Pattern Analysis and Applications*, 28, 1–15. https://doi.org/10.1007/s10044-024-01234-5

Ugale, M., Rodrigues, A., Shinde, O., Desle, K., & Yadav, S. (2023). A Review on Sign Language Recognition Using CNN. In *Proceedings of the International Conference on Applications of Machine Intelligence and Data Analytics (ICAMIDA 2022)* (pp. 251–259). Atlantis Press. https://doi.org/10.2991/978-94-6463-136-4_23

Undang-Undang Republik Indonesia Nomor 8 Tahun 2016 tentang Penyandang Disabilitas.

Universitas Prof. Dr. Hazairin, SH. (2024). Sistem Peringatan Dini Bencana Inklusif untuk Penyandang Tunarungu. *Jurnal Pengabdian Masyarakat Unihaz*, 5(2), 45–58.

Zhao, Z., & Chen, N.-Z. (2023). Spatial-Temporal Graph Convolutional Networks for Regression and Feature Extraction in Composite Structures. *Composite Structures*, 323, 117496. https://doi.org/10.1016/j.compstruct.2023.117496
