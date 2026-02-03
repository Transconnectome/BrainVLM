import re

def extract_label(text):
    # 찾고자 하는 패턴 정의
    # "appears to be a " 뒤에 오고 " subject." 앞에 오는 단어를 찾음
    pattern = r"appears to be a (\w+) subject\."
    
    match = re.search(pattern, text)
    
    if match:
        return match.group(1) # 찾은 단어 반환
    else:
        return None # 패턴이 없으면 None 반환




import json

# 파일 경로 설정 (실행하는 위치에 파일이 있어야 합니다)
input_file = '/Users/apple/Desktop/neuro-ai-research-system/projects/BrainVLM/code/BrainVLM-umbrella/UMBRELLA/predictions/sex_comparison_conversations_simple_extended_output_filtered.jsonl'

total = 0 
correct = 0 
wrong = 0 

subject_ids = [] 

print("필터링 작업을 시작합니다...")


try:
    with open(input_file, 'r', encoding='utf-8') as f_in:
        subject_id = None
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line: continue  # 빈 줄은 건너뜀

            try:
                data = json.loads(line)
                
                # JSON 구조에 맞춰 subject_ids 리스트 추출
                # 구조: root -> metadata -> sample_metadata -> subject_ids
                
                # 안전하게 키에 접근 (KeyError 방지)
                subject_id_tmp = data['metadata']['sample_metadata'].get('subject_id', None)
                label = data['metadata']['sample_metadata'].get('subject_label', [])
                generated_text = data.get('model_answer', [])
                pred = extract_label(generated_text)
                subject_ids.append(subject_id_tmp)
                if i == 0: 
                    subject_id = subject_id_tmp


                if subject_id_tmp == subject_id: 
                    pass 
                else: 
                    if label == pred: 
                        correct += 1 
                    else: 
                        print(f"SubjectID: {subject_id}, LABEL: {label}, PRED: {pred}")
                        wrong += 1 
                    total +=1
                    subject_id = subject_id_tmp
            
            except json.JSONDecodeError:
                print(f"JSON 형식이 잘못된 라인이 있어 건너뜁니다: {line[:50]}...")

    print("-" * 30)
    print(f"작업 완료!")
    print(f"Total: {total}, Correct: {correct}, Wrong: {wrong}, ACC: {(correct/total)}")
    import numpy as np 
    print(f"Total Subject: {len(np.unique(subject_ids,return_counts=True)[0])}")


except FileNotFoundError:
    print(f"오류: '{input_file}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")