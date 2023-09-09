import base64
import streamlit as st
from PIL import Image

icon = Image.open("img.png")
st.set_page_config(
    page_title="Fashion Finesse: Your Personalized Fashion Curator",
    page_icon=icon,
    layout="wide",
)

st.write("# Welcome to Fashion Finesse, Your Personalized Fashion Curator! 👋")
file_ = open("women_gif.gif", "rb")
contents = file_.read()
womendata_url = base64.b64encode(contents).decode("utf-8")
file_.close()
file_men = open("men_gif.gif", "rb")
contents_men = file_men.read()
mendata_url = base64.b64encode(contents_men).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{womendata_url}" alt="cat gif">'+
    f'<img src="data:image/gif;base64,{mendata_url}" alt="cat gif">'+
    """
     
    
Elevate Your Style with your personal fashion stylist!

Unlock the future of fashion with our cutting-edge AI Fashion Stylist. Discover the perfect outfit, redefine your wardrobe, and enhance your personal style effortlessly. Say goodbye to fashion dilemmas and hello to confidence in every step you take.

Why Choose AI Fashion Stylist?

1. **Personalized Style Recommendations**
   Our AI understands your unique fashion preferences, body type, and lifestyle. It curates personalized outfit suggestions tailored just for you. Whether you're dressing for a casual day out, a formal event, or anything in between, our AI has you covered.

2. **Stay Trendy and Time-Efficient**
   Stay ahead of the fashion curve without spending hours scrolling through endless options online or flipping through fashion magazines. Our AI keeps you up-to-date with the latest trends, so you can look your best without the hassle.

3. **Fashion for All Occasions**
   From daily attire to special occasions, our AI can guide you on what to wear. Whether it's a job interview, date night, or a weekend getaway, you'll always step out in style.

How it Works:

1. **Chat with Stella Stylebot and receive Personalized Suggestions**
   Instantly receive outfit recommendations based on your profile and wardrobe. You can mix and match items or get inspiration for new purchases.

2. **Stay Inspired**
   Keep up with the latest trends and style tips through our blog and newsletter. Learn how to accessorize, layer, and create stunning outfits that reflect your personality.

Why wait? Transform your fashion game with AI Fashion Stylist today. Join the fashion revolution and redefine your style effortlessly. It's time to look and feel your best, every day.

Ready to embark on your fashion journey? Create your profile and let our AI Fashion Stylist guide you to a more stylish and confident you.
""",
    unsafe_allow_html=True,
)