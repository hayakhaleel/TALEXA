from __future__ import annotations

from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"


def configure_page(page_title: str) -> None:
    st.set_page_config(page_title=page_title, layout="wide", initial_sidebar_state="collapsed")


def local_css(file_name: str = "style.css") -> None:
    with open(BASE_DIR / file_name, encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


def get_robot_image() -> Path | None:
    preferred_image = ASSETS_DIR / "robot.png"
    if preferred_image.exists():
        return preferred_image

    png_files = sorted(ASSETS_DIR.glob("*.png"))
    if png_files:
        return png_files[0]

    return None


def render_shell(title_text: str):
    local_css()

    st.markdown(
        """
        <div style="margin:0 0 20px 20px;">
            <h1 style="
                color:#1b1f8f;
                font-family: Arial Black, Arial, sans-serif;
                font-size:300px;
                margin:0;
                line-height:1;
            ">TALEXA</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<div style="height: 890px;"></div>', unsafe_allow_html=True)
        robot_image = get_robot_image()
        if robot_image:
            st.image(str(robot_image), width=2600)
        else:
            st.warning("Add a PNG image to the assets folder to show the robot illustration.")

    with right_col:
        st.markdown('<div style="height: 600px;"></div>', unsafe_allow_html=True)
        form_left, form_mid, form_right = st.columns([2.1, 5.1, 2.1])
        with form_mid:
            st.markdown(
                f'<div style="color:#000000; font-family:\'Times New Roman\', serif; '
                f'font-size:4.3125rem; font-weight:bold; margin-bottom:24px;">{title_text}</div>',
                unsafe_allow_html=True,
            )
            return st.container()

    return st.container()


def restore_auth_state() -> None:
    return None


def persist_auth_state(user_id: int | None, user_email: str | None) -> None:
    if user_id and user_email:
        st.query_params["user_id"] = str(user_id)
        st.query_params["user_email"] = user_email
    else:
        st.query_params.clear()


def get_auth_user_id() -> int | None:
    user_id = st.query_params.get("user_id")
    if not user_id:
        return None
    try:
        return int(user_id)
    except ValueError:
        return None


def get_auth_user_email() -> str | None:
    user_email = st.query_params.get("user_email")
    return str(user_email) if user_email else None
