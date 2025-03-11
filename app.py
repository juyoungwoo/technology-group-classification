import streamlit as st
import pandas as pd
import time
import squarify
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
import openai
import re
import matplotlib.font_manager as fm

# 한글 폰트 설정
def set_korean_font():
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우용
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

set_korean_font()

def fetch_product_simple_keywords(title, api_key):
    client = openai.OpenAI(api_key=api_key)  # ✅ 최신 방식으로 변경

    prompt = f"""
    This is the title of an invented patent: "{title}"
    Please suggest exactly 5 technologies that are highly relevant to this patent.
    Each technology name should be at most 2 words.
    Do not include explanations—only list the names.
    Response format: 'technology1, technology2, technology3, technology4, technology5'
    """
    
    try:
        time.sleep(0.1)

        response = client.chat.completions.create(  # ✅ 최신 OpenAI API 방식
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )

        result = response.choices[0].message.content.strip()  # ✅ 최신 방식으로 변경

        return result
    except Exception as e:
        st.error(f"Error fetching technologies: {e}")
        return ""


def process_text(text):
    cleaned_text = re.sub(r".*제품 카테고리:\s*", "", text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\d+\.\s?', '', cleaned_text)
    cleaned_text = cleaned_text.replace(' and ', ', ')
    cleaned_text = cleaned_text.replace('\n', ', ').strip()
    cleaned_text = cleaned_text.replace(' technology', 'technology')
    return cleaned_text

def count_word_frequency(dataframe, column_name):
    column_data = dataframe[column_name].dropna()
    all_words = [word.strip() for cell in column_data for word in set(str(cell).split(',')) if word.strip()]
    word_counts = Counter(all_words)
    return pd.DataFrame(word_counts.items(), columns=['technology', 'frequency']).sort_values(by='frequency', ascending=False)

def plot_treemap(data):
    if data.empty:
        st.warning("No technology data available for visualization.")
        return

    top_nouns = dict(data.head(25).set_index('technology')['frequency'])
    
    if not top_nouns:
        st.warning("No valid technology keywords to display in the treemap.")
        return
    
    norm = mpl.colors.Normalize(vmin=min(top_nouns.values()), vmax=max(top_nouns.values()))
    colors = [mpl.cm.Greens(norm(value)) for value in top_nouns.values()]
    labels = [f"{word}\n({freq})" for word, freq in top_nouns.items()]  # ✅ 줄바꿈 추가

    # 🔥 ✅ 트리맵 크기 자동 조정
    fig, ax = plt.subplots(figsize=(30, 15))  # 🔥 크기 확장
    
    # ✅ squarify의 padding 추가 → 글자가 겹치지 않도록 함
    squarify.plot(
        sizes=list(top_nouns.values()), 
        label=labels, 
        color=colors, 
        alpha=0.7, 
        text_kwargs={'fontsize': 12},  # ✅ 폰트 크기 조정
        ax=ax 
    )

    ax.set_title("Technology (Top 25)", fontsize=18, fontweight='bold')
    ax.axis('off')  # ✅ 축 제거

    st.pyplot(fig)  # ✅ 변경된 fig를 Streamlit에 표시


def main():
    st.title("Patent Technology Analysis")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload CSV file with '발명명칭' column", type=["csv"])
    
    if api_key and uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if '발명명칭' not in df.columns:
            st.error("CSV must contain '발명명칭' column.")
            return
        
        st.write("Processing patent titles...")
        df['기술'] = df['발명명칭'].apply(lambda title: fetch_product_simple_keywords(title, api_key))
        df['keywords'] = df['기술'].apply(process_text)
        
        # ✅ Extracted Technology Keywords 표시 및 다운로드 버튼 추가
        st.write("### Extracted Technology Keywords")
        st.dataframe(df[['발명명칭', 'keywords']])
        
        extracted_csv = df[['발명명칭', 'keywords']].to_csv(index=False, encoding='utf-8-sig')
        st.download_button("Download Extracted Technology Keywords CSV", data=extracted_csv, file_name="extracted_technology_keywords.csv", mime="text/csv")

        # ✅ Technology Frequency 테이블 및 다운로드 버튼 유지
        tech_freq = count_word_frequency(df, 'keywords')
        st.write("### Technology Frequency")
        st.dataframe(tech_freq)
        
        tech_freq_csv = tech_freq.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("Download Technology Frequency CSV", data=tech_freq_csv, file_name="technology_frequency.csv", mime="text/csv")
        
        st.write("### Technology Treemap")
        plot_treemap(tech_freq)


if __name__ == "__main__":
    main()
