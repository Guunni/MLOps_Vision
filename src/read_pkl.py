import pickle
import sys
from pathlib import Path
import json
from pprint import pprint

def analyze_pkl_file(pkl_path):
    """pkl 파일을 분석하여 내용을 출력합니다"""
    try:
        print(f"\n{'='*60}")
        print(f"분석할 파일: {pkl_path}")
        print(f"파일 크기: {Path(pkl_path).stat().st_size:,} bytes")
        print(f"{'='*60}")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n🔍 최상위 데이터 타입: {type(data)}")
        
        if isinstance(data, dict):
            print(f"📋 딕셔너리 키 개수: {len(data)}")
            print(f"📋 모든 키 목록:")
            for i, key in enumerate(data.keys(), 1):
                print(f"  {i:2d}. {key}")
            
            print(f"\n📊 각 키의 값 타입과 내용:")
            for key, value in data.items():
                print(f"\n  🔑 '{key}':")
                print(f"     타입: {type(value)}")
                
                if isinstance(value, (str, int, float, bool)):
                    print(f"     값: {value}")
                elif isinstance(value, (list, tuple)):
                    print(f"     길이: {len(value)}")
                    if len(value) <= 5:
                        print(f"     내용: {value}")
                    else:
                        print(f"     처음 3개: {value[:3]}...")
                elif isinstance(value, dict):
                    print(f"     키 개수: {len(value)}")
                    if len(value) <= 10:
                        print(f"     하위 키들: {list(value.keys())}")
                    else:
                        print(f"     하위 키들 (처음 5개): {list(value.keys())[:5]}...")
                elif hasattr(value, '__dict__'):
                    print(f"     객체 속성: {list(value.__dict__.keys()) if hasattr(value, '__dict__') else 'N/A'}")
                else:
                    print(f"     설명: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        
        elif isinstance(data, (list, tuple)):
            print(f"📋 {type(data).__name__} 길이: {len(data)}")
            if len(data) > 0:
                print(f"📋 첫 번째 요소 타입: {type(data[0])}")
                if len(data) <= 5:
                    for i, item in enumerate(data):
                        print(f"  {i}: {type(item)} - {str(item)[:50]}{'...' if len(str(item)) > 50 else ''}")
        
        else:
            print(f"📋 객체 타입: {type(data)}")
            if hasattr(data, '__dict__'):
                print(f"📋 객체 속성들: {list(data.__dict__.keys())}")
            print(f"📋 객체 설명: {str(data)[:200]}{'...' if len(str(data)) > 200 else ''}")
        
        print(f"\n{'='*60}")
        print("분석 완료")
        print(f"{'='*60}")
        
        return data
        
    except Exception as e:
        print(f"❌ 파일 분석 실패: {e}")
        return None

def main():
    pkl_files_dir = Path(__file__).parent.parent / "pkl_files"
    
    if not pkl_files_dir.exists():
        pkl_files_dir.mkdir()
        print(f"📁 {pkl_files_dir} 폴더를 생성했습니다.")
        print(f"   pkl 파일들을 이 폴더에 넣고 다시 실행하세요.")
        return
    
    # pkl 파일 목록 찾기
    pkl_files = list(pkl_files_dir.glob("*.pkl"))
    
    if not pkl_files:
        print(f"❌ {pkl_files_dir} 폴더에 pkl 파일이 없습니다.")
        return
    
    if len(pkl_files) == 1:
        # 파일이 하나면 바로 분석
        analyze_pkl_file(pkl_files[0])
    else:
        # 여러 파일이면 선택하게 함
        print(f"📁 발견된 pkl 파일들:")
        for i, file in enumerate(pkl_files, 1):
            print(f"  {i}. {file.name}")
        
        try:
            choice = input(f"\n분석할 파일 번호를 입력하세요 (1-{len(pkl_files)}): ")
            index = int(choice) - 1
            
            if 0 <= index < len(pkl_files):
                analyze_pkl_file(pkl_files[index])
            else:
                print("❌ 잘못된 번호입니다.")
                
        except ValueError:
            print("❌ 숫자를 입력해주세요.")

if __name__ == "__main__":
    main()