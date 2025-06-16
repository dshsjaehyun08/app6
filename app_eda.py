import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 페이지 설정
st.set_page_config(page_title="Population Trends EDA", layout="wide")

# 한글→영문 매핑
REGION_KR2EN = {
    "서울": "Seoul",    "부산": "Busan",   "대구": "Daegu",   "인천": "Incheon",
    "광주": "Gwangju",  "대전": "Daejeon", "울산": "Ulsan",   "세종": "Sejong",
    "경기": "Gyeonggi", "강원": "Gangwon", "충북": "Chungbuk","충남": "Chungnam",
    "전북": "Jeonbuk",  "전남": "Jeonnam", "경북": "Gyeongbuk","경남": "Gyeongnam",
    "제주": "Jeju",     "전국": "National"
}

def load_population_df(file_obj: io.BytesIO) -> pd.DataFrame:
    """CSV 업로드 → 전처리 일괄 수행"""
    df = pd.read_csv(file_obj)
    # 세종 지역 '-' → 0 치환
    mask = df["지역"] == "세종"
    df.loc[mask, ["인구", "출생아수(명)", "사망자수(명)"]] = (
        df.loc[mask, ["인구", "출생아수(명)", "사망자수(명)"]]
          .replace("-", 0)
    )
    # 숫자형으로 변환
    num_cols = ["인구", "출생아수(명)", "사망자수(명)"]
    df[num_cols] = df[num_cols] \
        .apply(pd.to_numeric, errors="coerce") \
        .fillna(0) \
        .astype(int)
    # 영문 지역명 컬럼 추가
    df["region_en"] = df["지역"].map(REGION_KR2EN)
    return df

def predict_pop_2035(nat_df: pd.DataFrame) -> int:
    """최근 3개년 출생-사망 평균으로 2035년 인구 예측"""
    recent = nat_df.sort_values("연도").tail(3)
    annual_delta = int((recent["출생아수(명)"] - recent["사망자수(명)"]).mean())
    years = 2035 - recent["연도"].max()
    return int(nat_df.iloc[-1]["인구"] + annual_delta * years)

class EDA:
    def __init__(self):
        st.title("📊 Exploratory Data Analysis")
        st.markdown("## 🗂️ Population Trends (새 과제)")
        pop_file = st.file_uploader("population_trends.csv 업로드", type="csv")

        if pop_file:
            pop_df = load_population_df(pop_file)
            tabs = st.tabs([
                "Basic Stats", "National Trend", "Regional Δ (5y)",
                "Top Δ records", "Heatmap / Area"
            ])

            # 1) Basic Stats
            with tabs[0]:
                st.subheader("📄 Data Overview")
                buf = io.StringIO()
                pop_df.info(buf=buf)
                st.text(buf.getvalue())
                st.dataframe(pop_df.describe())

            # 2) National Trend
            with tabs[1]:
                nat = pop_df[pop_df["지역"] == "전국"].sort_values("연도")
                fig, ax = plt.subplots()
                ax.plot(nat["연도"], nat["인구"], marker="o", label="Actual")
                forecast = predict_pop_2035(nat)
                ax.plot(
                    [nat["연도"].max(), 2035],
                    [nat["인구"].iloc[-1], forecast],
                    linestyle="--", marker="^", label="Forecast 2035"
                )
                ax.set_xlabel("Year")
                ax.set_ylabel("Population")
                ax.legend()
                st.pyplot(fig)

            # 3) Regional Δ (5y)
            with tabs[2]:
                last_year = pop_df["연도"].max()
                base = pop_df[pop_df["연도"] == last_year - 4].set_index("지역")["인구"]
                curr = pop_df[pop_df["연도"] == last_year].set_index("지역")["인구"]
                delta = (curr - base).drop("전국").sort_values(ascending=False)

                fig, ax = plt.subplots(figsize=(7, 6))
                sns.barplot(
                    x=delta.values/1000,
                    y=[REGION_KR2EN[k] for k in delta.index],
                    orient="h",
                    ax=ax
                )
                for i, v in enumerate(delta.values/1000):
                    ax.text(v, i, f"{v:,.0f}", va="center")
                ax.set_xlabel("Δ (× 1,000)")
                st.pyplot(fig)

                st.markdown(
                    f"**Top 3 상승 지역:** {', '.join(delta.head(3).index)}\n"
                    f"**Top 3 감소 지역:** {', '.join(delta.tail(3).index)}"
                )

            # 4) Top Δ records
            with tabs[3]:
                diff_df = (
                    pop_df.sort_values(["지역", "연도"])
                          .assign(diff=lambda d: d.groupby("지역")["인구"].diff())
                          .query("지역 != '전국'")
                )
                top100 = diff_df.nlargest(100, "diff")
                styled = (
                    top100.style
                         .format({"diff": "{:,+}"})
                         .background_gradient(
                             subset=["diff"],
                             vmin=-top100["diff"].abs().max(),
                             vmax= top100["diff"].abs().max(),
                             cmap="coolwarm"
                         )
                )
                st.dataframe(styled, use_container_width=True)

            # 5) Heatmap / Area
            with tabs[4]:
                pivot = (
                    pop_df.pivot_table(index="region_en", columns="연도", values="인구")
                          .loc[lambda d: d.index != "National"]
                )
                st.subheader("Heatmap")
                fig_h, ax_h = plt.subplots(figsize=(10, 6))
                sns.heatmap(pivot/1000, cmap="YlGnBu", ax=ax_h)
                ax_h.set_xlabel("Year")
                ax_h.set_ylabel("Region")
                st.pyplot(fig_h)

                st.subheader("Stacked Area")
                nat_area = pop_df[pop_df["지역"] != "전국"] \
                             .pivot_table(index="연도", columns="region_en", values="인구", aggfunc="sum")
                fig_a, ax_a = plt.subplots(figsize=(10, 4))
                ax_a.stackplot(nat_area.index, nat_area.T/1000, labels=nat_area.columns)
                ax_a.legend(ncol=4, fontsize="small")
                ax_a.set_xlabel("Year")
                ax_a.set_ylabel("Population (× 1,000)")
                st.pyplot(fig_a)

def main():
    EDA()

if __name__ == "__main__":
    main()
