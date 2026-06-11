"""
AUTSL Dataset Decryption Script
İndirdiğiniz şifreli dosyaları açar ve kullanıma hazır hale getirir.
"""

import os
import subprocess
import zipfile
from pathlib import Path

# Decryption keys
KEYS = {
    'train': 'MdG3z6Eh1t',
    'validation': 'bhRY5B9zS2',
    'validation_labels': 'zYX5W7fZ',
    'test': 'ds6Kvdus3o'
}

# Klasör yapısı
ENCRYPTED_DIR = Path("autsl_encrypted")
OUTPUT_DIR = Path("autsl_original")

def decrypt_file(encrypted_file, decrypted_file, password):
    """
    Şifreli dosyayı açar.
    OpenSSL veya 7-Zip kullanır.
    """
    print(f"\n🔓 Decrypting: {encrypted_file.name}")
    
    # OpenSSL ile dene
    try:
        cmd = [
            'openssl', 'enc', '-d', '-aes-256-cbc',
            '-in', str(encrypted_file),
            '-out', str(decrypted_file),
            '-k', password,
            '-md', 'md5'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Decrypted successfully: {decrypted_file.name}")
            return True
    except FileNotFoundError:
        print("⚠️ OpenSSL bulunamadı, 7-Zip deneniyor...")
    
    # 7-Zip ile dene
    try:
        cmd = [
            '7z', 'x',
            f'-p{password}',
            str(encrypted_file),
            f'-o{decrypted_file.parent}',
            '-y'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Decrypted successfully with 7-Zip")
            return True
    except FileNotFoundError:
        print("❌ 7-Zip de bulunamadı!")
    
    print("❌ Decryption failed! OpenSSL veya 7-Zip kurulu değil.")
    return False

def extract_zip(zip_file, extract_dir):
    """ZIP dosyasını çıkartır."""
    print(f"\n📦 Extracting: {zip_file.name}")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Extracted to: {extract_dir}")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def main():
    """Ana işlem."""
    print("="*60)
    print("AUTSL Dataset Decryption & Extraction")
    print("="*60)
    
    # Klasörleri oluştur
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if not ENCRYPTED_DIR.exists():
        print(f"\n❌ Hata: {ENCRYPTED_DIR} klasörü bulunamadı!")
        print(f"📥 Lütfen indirdiğiniz dosyaları {ENCRYPTED_DIR} klasörüne koyun.")
        return
    
    encrypted_files = list(ENCRYPTED_DIR.glob("*"))
    if not encrypted_files:
        print(f"\n⚠️ {ENCRYPTED_DIR} klasörü boş!")
        print(f"📥 İndirdiğiniz dosyaları buraya koyun:")
        print(f"   - train_data.zip (veya .tar.gz, .7z)")
        print(f"   - validation_data.zip")
        print(f"   - validation_labels.csv")
        print(f"   - test_data.zip")
        print(f"   - test_labels.csv")
        return
    
    print(f"\n📂 Bulunan dosyalar:")
    for f in encrypted_files:
        print(f"   - {f.name} ({f.stat().st_size / (1024**3):.2f} GB)")
    
    print("\n" + "="*60)
    print("İşlem başlıyor...")
    print("="*60)
    
    # Her dosyayı işle
    for enc_file in encrypted_files:
        if enc_file.is_dir():
            continue
            
        # Dosya tipini belirle
        file_type = None
        key = None
        
        if 'train' in enc_file.name.lower():
            file_type = 'train'
            key = KEYS['train']
        elif 'validation' in enc_file.name.lower() and 'label' not in enc_file.name.lower():
            file_type = 'validation'
            key = KEYS['validation']
        elif 'validation' in enc_file.name.lower() and 'label' in enc_file.name.lower():
            file_type = 'validation_labels'
            key = KEYS['validation_labels']
        elif 'test' in enc_file.name.lower():
            file_type = 'test'
            key = KEYS['test']
        
        if not file_type or not key:
            print(f"\n⚠️ Atlanan dosya (tip belirlenemedi): {enc_file.name}")
            continue
        
        # Decrypt edilmiş dosya adı
        dec_filename = enc_file.stem  # .enc uzantısını kaldır
        if not dec_filename.endswith(('.zip', '.tar', '.gz', '.csv')):
            dec_filename += '.zip'
        
        dec_file = OUTPUT_DIR / dec_filename
        
        # Decrypt et
        if dec_file.exists():
            print(f"\n⏭️ Zaten decrypt edilmiş: {dec_file.name}")
        else:
            success = decrypt_file(enc_file, dec_file, key)
            if not success:
                continue
        
        # ZIP ise extract et
        if dec_file.suffix == '.zip':
            extract_dir = OUTPUT_DIR / file_type
            extract_dir.mkdir(exist_ok=True)
            extract_zip(dec_file, extract_dir)
    
    print("\n" + "="*60)
    print("✅ İşlem tamamlandı!")
    print("="*60)
    print(f"\n📁 Dosyalar buraya çıkartıldı: {OUTPUT_DIR.absolute()}")
    print("\n📋 Sonraki adım:")
    print("   python prepare_autsl_videos.py")

if __name__ == "__main__":
    main()
