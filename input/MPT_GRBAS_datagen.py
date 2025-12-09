import pandas as pd
import os
import glob

def create_comprehensive_dataset():
    print("데이터셋 제작을 시작합니다...")

    target_vowels = ['001', '007', '008'] 
    excel_path = 'GRBAS_data.xlsx'
    
    sheets_config = [
        {'idx': 0, 'prefix': 'SLP1'},
        {'idx': 1, 'prefix': 'SLP2'},
        {'idx': 2, 'prefix': 'SLP3'},
        {'idx': 3, 'prefix': 'SLPall'}
    ]
    
    clinical_df = None

    try:
        for config in sheets_config:
            df = pd.read_excel(excel_path, sheet_name=config['idx'], header=0)
            
            id_col = df.columns[0] 
            df = df.rename(columns={id_col: 'patient_ID'})
            df['patient_ID'] = df['patient_ID'].astype(str)
            
            score_cols = df.columns[1:] 
            new_names = {col: f"{config['prefix']}_{col}" for col in score_cols}
            df = df.rename(columns=new_names)
            
            if clinical_df is None:
                clinical_df = df
            else:
                clinical_df = pd.merge(clinical_df, df, on='patient_ID', how='outer')
                
        print(f"엑셀 데이터 로드 완료: 총 {len(clinical_df)}명의 환자 정보")
        
    except Exception as e:
        print(f"엑셀 파일 처리 중 오류 발생: {e}")
        return

    wav_root_dir = '/home/nas4/DB/DB_XAI_v1.9.2/v1.9.2/wav'
    audio_records = []

    wav_files = glob.glob(os.path.join(wav_root_dir, '**', '*.wav'), recursive=True)
    print(f"총 {len(wav_files)}개의 wav 파일 탐색 중...")

    for file_path in wav_files:
        filename = os.path.basename(file_path)
        file_stem = os.path.splitext(filename)[0]
        parts = file_stem.split('_')
        
        if len(parts) >= 3:
            p_id = parts[0]
            vowel = parts[1]

            if vowel in target_vowels:
                audio_records.append({
                    'patient_ID': p_id,
                    'vowel_label': vowel,
                    'wav_path': file_path
                })
            else:
                pass
        else:
            print(f"형식 불일치 파일: {filename}")

    if not audio_records:
        print("조건에 맞는 wav 파일을 찾지 못했습니다. 모음 라벨이나 파일 경로를 확인해주세요.")
        return

    audio_df = pd.DataFrame(audio_records)
    print(f"필터링 후 오디오 데이터: {len(audio_df)}행")

    final_df = pd.merge(audio_df, clinical_df, on='patient_ID', how='inner')
    final_df = final_df.sort_values(by=['patient_ID', 'vowel_label'], ascending=[True, True])
    
    cols = ['patient_ID', 'vowel_label', 'wav_path'] + [c for c in final_df.columns if c not in ['patient_ID', 'vowel_label', 'wav_path']]
    final_df = final_df[cols]

    output_filename = 'GRBAS_dataset.csv'
    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print(f"작업 완료! '{output_filename}' 파일 생성됨.")
    print(f"포함된 모음: {target_vowels}")
    print(f"최종 데이터 크기: {final_df.shape}")
    print("-" * 30)
    print(final_df.head())

if __name__ == "__main__":
    create_comprehensive_dataset()