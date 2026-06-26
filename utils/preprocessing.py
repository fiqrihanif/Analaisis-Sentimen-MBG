import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- 1. KAMUS NORMALISASI ---
normalization_dict = {
    'gak':'tidak','ga':'tidak','nggak':'tidak','ngga':'tidak','gk':'tidak',
    'tdk':'tidak','ndak':'tidak','udh':'sudah','udah':'sudah','dah':'sudah',
    'sdh':'sudah','bgt':'banget','bngt':'banget','sy':'saya','aq':'aku',
    'gue':'saya','gw':'saya','lu':'kamu','lo':'kamu','km':'kamu',
    'yg':'yang','utk':'untuk','dgn':'dengan','dr':'dari','dlm':'dalam',
    'tsb':'tersebut','scr':'secara','krn':'karena','karna':'karena',
    'bs':'bisa','bsa':'bisa','msh':'masih','blm':'belum','sdg':'sedang',
    'lg':'lagi','jg':'juga','tp':'tapi','ttg':'tentang','ny':'nya',
    'klo':'kalau','klu':'kalau','kl':'kalau','spt':'seperti','pd':'pada',
    'spy':'supaya','aja':'saja','aj':'saja','lah':'','deh':'','dong':'',
    'nih':'','wkwk':'','wkwkwk':'','haha':'','hehe':'','ok':'oke',
    'mkn':'makan','mskpn':'meskipun','krg':'kurang','sblm':'sebelum',
    'stlh':'setelah','bbrp':'beberapa','hrs':'harus','smua':'semua',
    'emang':'memang','emg':'memang','mgkn':'mungkin',
    'makasih':'terima kasih','terimakasih':'terima kasih','thx':'terima kasih',
}

# --- 2. CUSTOM STOPWORDS ---
custom_stopwords = {
    'mbg','makan','bergizi','gratis','program','prabowo',
    'indonesia','siswa','sekolah','anak','pemerintah','rt',
    'https','http','amp','via','co','twitter','tweet',
    'retweet','follow','like','share','baca','selengkapnya',
    'liputan','berita','foto','video','link','klik',
    'siang','siiang','pak','buat','jadi','sama','presiden',
    'rakyat','banyak','negara','kalau','apa','kerja','anggar',
    'gibran','subianto','purn','tni','jenderal','drs','listyo',
    'sigit','kapolri','inisiasi','nasional','lalu','langsung',
    'jakarta','wakil','layan','masyarakat','hari','orang',
    'kita','mereka','ada','akan','lebih','sudah','juga',
    'dalam','untuk','dengan','dari','tapi','memang','harus',
    'semua','masih','saja','biar','satu','dua','tiga',
}

def normalize_text(text):
    """Mengubah kata tidak baku menjadi baku berdasarkan normalization_dict"""
    tokens = text.split()
    normalized = [normalization_dict.get(t, t) for t in tokens]
    return ' '.join([t for t in normalized if t != ''])

def preprocess_input(text, stemmer, sastrawi_stopwords):
    """
    Fungsi utama untuk membersihkan teks input dari pengguna di Streamlit.
    Alurnya dibuat sama persis dengan yang ada di Jupyter Notebook.
    """
    # Gabungkan stopwords dari Sastrawi dan custom
    all_stopwords = sastrawi_stopwords | custom_stopwords
    
    # 1. Case Folding
    text = text.lower()
    
    # 2. Cleansing
    text = re.sub(r'http\S+|www\S+', '', text) # Hapus URL
    text = re.sub(r'@\w+', '', text)           # Hapus Mention
    text = re.sub(r'#(\w+)', r'\1', text)      # Hapus '#' tapi simpan kata di dalamnya
    text = re.sub(r'&\w+;', '', text)          # Hapus HTML entities
    text = text.encode('ascii', 'ignore').decode('ascii') # Hapus emoji / non-ascii
    text = re.sub(r'[^a-z\s]', '', text)       # Hapus angka dan tanda baca
    text = re.sub(r'\s+', ' ', text).strip()   # Hapus spasi berlebih
    
    # 3. Normalisasi
    text = normalize_text(text)
    
    # 4. Filter Stopwords (Putaran 1) & Filter panjang kata > 2
    tokens = text.split()
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
    
    # 5. Stemming (Sastrawi)
    tokens = [stemmer.stem(t) for t in tokens]
    
    # 6. Filter Stopwords (Putaran 2) - setelah stemming mungkin ada kata dasar yg jadi stopword
    tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
    
    return ' '.join(tokens)