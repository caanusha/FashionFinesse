import pickle
import random
import openai
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt  # plotting
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
import cv2
from PIL import Image

# Set the model engine and your OpenAI API key
model_engine = "text-davinci-003"
openai.api_key = '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'
icon = Image.open("img.png")
st.set_page_config(
    page_title="Fashion Finesse: Your Personalized Fashion Curator",
    page_icon=icon,
    layout="wide",
)
women_traditional = [226, 2065, 2441, 5685]
men_traditional = [1430, 5374, 2137, 5552]
formal_men = [51, 717, 814, 1143]
formal_women = [3797, 629, 159, 393]
casual_women = [26, 148, 438, 711]
casual_men = [5228, 3, 1, 258, 369]


def plot_figures(figures, nrows=1, ncols=1, figsize=(8, 8)):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()  # optionals
    plt.savefig('x', dpi=400)
    st.image('x.png')
    # return time


def img_path(img):
    return DATASET_PATH + "/images/" + img


def load_image(img, resized_fac=0.1):
    img = cv2.imread(img_path(img))
    w, h, _ = img.shape
    resized = cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_AREA)
    return resized


DATASET_PATH = "C:/Users/aishu/Downloads/fashion-dataset/"

df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=7103, on_bad_lines='skip')
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.reset_index(drop=True)

with open('fashion.pkl', 'rb') as f:
    fashion_embed = pickle.load(f)  # deserialize using load()

# Calcule DIstance Matriz
cosine_sim = 1 - pairwise_distances(fashion_embed, metric='cosine')

indices = pd.Series(range(len(df)), index=df.index)


# Function that get movie recommendations based on the cosine similarity score of movie genres
def get_recommender(idx, df, top_n=5):
    sim_idx = indices[idx]
    print(df.index[idx])
    sim_scores = list(enumerate(cosine_sim[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    idx_rec = [i[0] for i in sim_scores]
    idx_sim = [i[1] for i in sim_scores]

    return indices.iloc[idx_rec].index, idx_sim


st.title("Hello! I am Stella Style bot, How can I help you?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == 'image':
            st.image(message["content"])
        else:
            st.markdown(message["content"])


def ChatGPT(user_query):
    # Use the OpenAI API to generate a response
    #response = "xyz"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "assistant", "content": "You are a fashion stylist assistant."},
            {"role": "user", "content": user_query}
        ]
    )
    #return response + " Below are few examples which you can purchase on XYZ website"
    return response['choices'][0]['message']['content'] + " Below are few examples which you can purchase on XYZ website"


# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if "traditional" in prompt or "ethnic" in prompt or "conventional" in prompt:
            if "women" in prompt:
                print("trad-women")
                idx_ref = women_traditional[random.randint(0, (len(women_traditional) - 1))]
            else:
                print("trad-men")
                idx_ref = men_traditional[random.randint(0, (len(men_traditional) - 1))]
        elif "casual" in prompt or "informal" in prompt:
            if "women" in prompt:
                print("casual - women")
                idx_ref = casual_women[random.randint(0, (len(casual_women) - 1))]
            else:
                print("casual - men")
                idx_ref = casual_men[random.randint(0, (len(casual_men) - 1))]
        elif "formal" in prompt:
            if "women" in prompt:
                print("formal - women")
                idx_ref = formal_women[random.randint(0, (len(formal_women) - 1))]
            else:
                print("formal - men")
                idx_ref = formal_men[random.randint(0, (len(formal_men) - 1))]
        else:
            idx_ref = 0
        # Recommendations
        idx_rec, idx_sim = get_recommender(idx_ref, df, top_n=6)

        # Plot
        # ===================
        plt.imshow(cv2.cvtColor(load_image(df.iloc[idx_ref].image), cv2.COLOR_BGR2RGB))

        # generation of a dictionary of (title, images)
        print({'im' + str(i): load_image(row.image) for i, row in df.loc[idx_rec].iterrows()})
        figures = {'im' + str(i): load_image(row.image) for i, row in df.loc[idx_rec].iterrows()}

        response = ChatGPT(prompt)
        st.markdown(response)
        # plot of the images in a figure, with 2 rows and 3 columns
        plot_figures(figures, 2, 3)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.messages.append({"role": "image", "content": 'x.png'})
