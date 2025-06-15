# ──────────────────────────────────────────────────────────
# 0) 공통 import
# ──────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, time, textwrap

# (Firebase · 로그인/회원가입 영역은 생략)
# …

# ──────────────────────────────────────────────────────────
# ✨ 1) Population 전용 도우미 함수
# ──────────────────────────────────────────────────────────
REGION_KR2EN = {
    "서울":"Seoul","부산":"Busan","대구":"Daegu","인천":"Incheon","광주":"Gwangju",
    "대전":"Daejeon","울산":"Ulsan","세종":"Sejong","경기":"Gyeonggi","강원":"Gangwon",
    "충북":"Chungbuk","충남":"Chungnam","전북":"Jeonbuk","전남":"Jeonnam","경북":"Gyeongbuk",
    "경남":"Gyeongnam","제주":"Jeju","전국":"National"
}
def load_population_df(file_obj: io.BytesIO) -> pd.DataFrame:
    """CSV 업로드 → 전처리 일괄 수행"""
    df = pd.read_csv(file_obj)
    # ① 세종 '-' → 0 치환
    mask_sejong = df["지역"] == "세종"
    df.loc[mask_sejong, ["인구", "출생아수(명)", "사망자수(명)"]] = (
        df.loc[mask_sejong, ["인구", "출생아수(명)", "사망자수(명)"]].replace("-", 0)
    )
    # ② 문자열 → 숫자
    num_cols = ["인구", "출생아수(명)", "사망자수(명)"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    # ③ 영문 지역명 추가 (그래프용)
    df["region_en"] = df["지역"].map(REGION_KR2EN)
    return df

def predict_pop_2035(nat_df: pd.DataFrame) -> int:
    """최근 3개년 출생–사망 평균으로 2035년 인구 간이 예측"""
    recent = nat_df.sort_values("연도").tail(3)
    annual_delta = int((recent["출생아수(명)"] - recent["사망자수(명)"]).mean())
    yrs = 2035 - recent["연도"].max()
    return int(nat_df.iloc[-1]["인구"] + annual_delta * yrs)

# ──────────────────────────────────────────────────────────
# 2) EDA 페이지 클래스 (기존 + Population EDA 탭 추가)
# ──────────────────────────────────────────────────────────
class EDA:
    def __init__(self):
        st.title("📊 Exploratory Data Analysis")

        # ===== 2‑A) 기존 Bike Sharing EDA 영역 =====
        # (원본 8 개 탭 그대로 ― 코드 생략) …

        # ===== 2‑B) Population EDA 영역 =====
        # --------------------------------------------------
        # **변경: 기존 8 개 탭 배열 뒤에 새 항목을 추가**
        # --------------------------------------------------
        st.markdown("## 🗂️ Population Trends (새 과제)")
        pop_file = st.file_uploader("`population_trends.csv` 업로드", type="csv")

        if pop_file:
            pop_df = load_population_df(pop_file)

            # 상위 탭 구성
            pop_tabs = st.tabs([
                "Basic Stats", "National Trend", "Regional Δ (5y)",
                "Top Δ records", "Heatmap / Area"
            ])
            # ─────────────────────────────
            # ① Basic Stats
            # ─────────────────────────────
            with pop_tabs[0]:
                st.subheader("📄 Data Overview")
                buf = io.StringIO(); pop_df.info(buf=buf)
                st.text(buf.getvalue())
                st.dataframe(pop_df.describe())

            # ─────────────────────────────
            # ② National Trend
            # ─────────────────────────────
            with pop_tabs[1]:
                nat = pop_df[pop_df["지역"] == "전국"].sort_values("연도")
                fig, ax = plt.subplots()
                ax.plot(nat["연도"], nat["인구"], marker="o", label="Actual")
                # 2035 예측치
                pop2035 = predict_pop_2035(nat)
                ax.plot([nat["연도"].max(), 2035],
                        [nat["인구"].iloc[-1], pop2035],
                        linestyle="--", marker="^", label="Forecast 2035")
                ax.set_xlabel("Year"); ax.set_ylabel("Population"); ax.legend()
                st.pyplot(fig)

            # ─────────────────────────────
            # ③ Regional Δ (최근 5년)
            # ─────────────────────────────
            with pop_tabs[2]:
                latest_year = pop_df["연도"].max()
                base_year = latest_year - 4
                base = pop_df[pop_df["연도"] == base_year] \
                          .set_index("지역")["인구"]
                latest = pop_df[pop_df["연도"] == latest_year] \
                            .set_index("지역")["인구"]
                delta = (latest - base).drop("전국").sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(7,6))
                sns.barplot(x=delta.values/1000, y=[REGION_KR2EN[k] for k in delta.index],
                            orient="h", ax=ax)
                for i,(v,k) in enumerate(zip(delta.values/1000, delta.index)):
                    ax.text(v, i, f"{v:,.0f}", va='center')
                ax.set_xlabel("Δ (× 1,000)")
                st.pyplot(fig)

                st.markdown(
                    f"**Top 3 상승 지역:** {', '.join(delta.head(3).index)}  \n"
                    f"**Top 3 감소 지역:** {', '.join(delta.tail(3).index)}"
                )

            # ─────────────────────────────
            # ④ Top Δ records (연도별 증감 상위 100)
            # ─────────────────────────────
            with pop_tabs[3]:
                diff_df = (
                    pop_df.sort_values(["지역","연도"])
                          .assign(diff=lambda d: d.groupby("지역")["인구"].diff())
                          .query("지역 != '전국'")
                )
                top100 = diff_df.nlargest(100, "diff").copy()
                # 숫자 포맷 & 컬러바
                styled = (
                    top100.style
                    .format({"diff": "{:,+}"})
                    .background_gradient(subset=["diff"],
                                         vmin=-top100["diff"].abs().max(),
                                         vmax= top100["diff"].abs().max(),
                                         cmap="coolwarm")
                )
                st.dataframe(styled, use_container_width=True)

            # ─────────────────────────────
            # ⑤ Heatmap / Stacked Area
            # ─────────────────────────────
            with pop_tabs[4]:
                pivot = (
                    pop_df.pivot_table(index="region_en",
                                       columns="연도",
                                       values="인구")
                    .loc[lambda d: d.index != "National"]  # 전국 제외
                )
                # 🔹 Heatmap
                st.subheader("Heatmap")
                fig_h, ax_h = plt.subplots(figsize=(10,6))
                sns.heatmap(pivot/1_000, cmap="YlGnBu", ax=ax_h)
                ax_h.set_xlabel("Year"); ax_h.set_ylabel("Region")
                st.pyplot(fig_h)

                # 🔹 Stacked Area
                st.subheader("Stacked Area (National)")
                nat_area = pop_df[pop_df["지역"] != "전국"] \
                              .pivot_table(index="연도",
                                           columns="region_en",
                                           values="인구",
                                           aggfunc="sum")
                fig_a, ax_a = plt.subplots(figsize=(10,4))
                ax_a.stackplot(nat_area.index, nat_area.T/1_000,
                               labels=nat_area.columns)
                ax_a.legend(ncol=4, fontsize="small")
                ax_a.set_xlabel("Year"); ax_a.set_ylabel("Pop (× 1,000)")
                st.pyplot(fig_a)

# (페이지 등록부는 그대로)
# ──────────────────────────────────────────────────────────
