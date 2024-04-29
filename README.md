# Machine-Learning
### Machine Learning using Python 
파이썬을 사용한 기계학습 (Python 3.8.10)
<br/><br/>

## 실행을 위해 필요한 주요 패키지 <br/>
해당 버전이 없거나 오류 발생 시 다른 버전, 최신 버전 사용하기
|패키지|버전|패키지|버전|
|:-|:-|:-|:-|
|librosa|0.8.1|plotly|5.3.1|
|matplotlib|3.4.3|scikit-learn|1.1.3|
|numba|0.53.1|scipy|1.7.1|
|numpy|1.21.2|seaborn|0.11.2|
|pandas|1.3.2|sklearn|0.0|

<br/>

## 패키지 설치 가이드 <br/>
### cmd(명령 프롬프트)창에서...
- 파일 존재 확인하기 "pip show (파일명)" <br/>
- 원하는 버전 설치하기 "pip install (파일명)==x.xx.x" <br/>
```
pip install librosa==0.8.1 <br/>
pip show librosa <br/>
```
### pip 명령어가 안되면... 
- pip 대신 pip3 사용하기 <br/>
- 파이썬 설치 여부 확인하기 <br/>
- 해당 버전이 존재하지 않을 때 : pip install (파일명) <br/>
- 환경변수 설정하기 : https://hungdung99.tistory.com/9 <br/>
- 오류메세지 `No module named 'pip'` : https://puleugo.tistory.com/18 <br/>
```
pip3 install librosa==0.8.1 <br/>
pip3 install librosa <br/>
pip install librosa <br/>
```
### 맥OS에서 구동 실패 시, 패키지 최신 버전 다운 <br/>
```
pip install numba --upgrade <br/>
pip install matplotlib --upgrade <br/>
```
