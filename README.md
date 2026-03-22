# Data Science & MLOps Learning Resources

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](#contributing)

A curated collection of tutorials, notebooks, and reference materials for learning **data science** and **machine learning operations (MLOps)** — from the fundamentals all the way to advanced topics like generative AI.

Whether you are a complete beginner curious about what machine learning is, or a practicing engineer looking to sharpen your MLOps skills, you will find practical, hands-on resources here to help you on your journey.

> **Note:** These materials are designed to introduce key concepts and provide hands-on practice. They are not intended to be an exhaustive manual — think of them as a guided starting point.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Environment Setup](#environment-setup)
- [Beginner Resources](#beginner-resources)
- [Foundational Skills](#foundational-skills)
  - [SQL](#sql)
  - [Mathematics & Statistics](#mathematics--statistics)
  - [Python Programming](#python-programming)
- [Data Science](#data-science)
  - [Data Science Life Cycle](#data-science-life-cycle)
  - [Data Wrangling & Feature Engineering](#data-wrangling--feature-engineering)
  - [Data Visualization](#data-visualization)
- [Machine Learning](#machine-learning)
  - [Classical Machine Learning](#classical-machine-learning)
  - [Deep Learning](#deep-learning)
- [Natural Language Processing (NLP)](#natural-language-processing-nlp)
- [Generative AI & Large Language Models](#generative-ai--large-language-models)
  - [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Prompt Engineering](#prompt-engineering)
- [Reinforcement Learning](#reinforcement-learning)
- [Explainability & Data-Centric AI](#explainability--data-centric-ai)
- [MLOps Platforms](#mlops-platforms)
- [Video Courses & Talks](#video-courses--talks)
- [Books & Extended Reading](#books--extended-reading)
- [Contributing](#contributing)

---

## Getting Started

To download a local copy of this repository, install [Git](https://git-scm.com/downloads) and run:

```shell
git clone https://github.com/chen115y/MLOpsLearning.git
```

New to Git and GitHub? Check out this [introductory guide](./Python_Introduction/git_github.pdf) included in the repository.

---

## Environment Setup

You will need a Python environment to run the notebooks. Choose whichever option best fits your situation:

| Option | Best For | Link |
|--------|----------|------|
| **Local setup (Windows / Ubuntu)** | Full control over your environment | [Environment Setup Guide](https://github.com/chen115y/DataScience_Env_Setup) |
| **Google Colab** | Quick start with zero installation | [Open Colab](https://colab.research.google.com/notebooks/intro.ipynb) |
| **Anaconda** | Pre-packaged data-science libraries | [Download Anaconda](https://www.anaconda.com/distribution/) |

**Recommended editors / IDEs:**

- [Visual Studio Code](https://code.visualstudio.com/docs/python/jupyter-support) — lightweight, with excellent Jupyter support
- [JupyterLab / Jupyter Notebook](https://jupyter.org/) — the classic notebook interface
- [PyCharm](https://www.jetbrains.com/pycharm/) — full-featured Python IDE

---

## Beginner Resources

Brand-new to AI and data science? These free courses are a great place to start — no prior experience required.

- [Generative AI for Beginners](https://microsoft.github.io/generative-ai-for-beginners/#/) — Microsoft's hands-on curriculum covering generative AI fundamentals
- [Machine Learning for Beginners](https://aka.ms/ml-beginners?WT.mc_id=academic-105485-koreyst) — a 12-week, 26-lesson curriculum by Microsoft
- [Data Science for Beginners](https://aka.ms/datascience-beginners?WT.mc_id=academic-105485-koreyst) — a 10-week, 20-lesson curriculum by Microsoft
- [AI for Beginners](https://aka.ms/ai-beginners?WT.mc_id=academic-105485-koreyst) — a 12-week, 24-lesson curriculum by Microsoft

---

## Foundational Skills

Before diving into machine learning, it helps to be comfortable with a few building blocks: querying data with SQL, core math and statistics, and Python programming.

### SQL

SQL (Structured Query Language) is the standard language for working with databases — a must-know for any data professional.

- [SQL Essentials for Beginners](https://www.udemy.com/course/sql-essentials-for-beginners/) — Udemy course covering the fundamentals

### Mathematics & Statistics

You do not need a math degree, but a working knowledge of probability, statistics, and linear algebra will go a long way.

- [Khan Academy — Math](https://www.khanacademy.org/math) — free, self-paced lessons from arithmetic to linear algebra
- [Hypothesis Tests in One Picture](https://www.datasciencecentral.com/profiles/blogs/hypothesis-tests-in-one-picture) — a handy visual reference
- [Think Stats: Probability and Statistics for Programmers](./Python_Introduction/thinkstats.pdf) — a free, code-first textbook
- [Mathematics for Machine Learning](https://mml-book.github.io/) — a comprehensive, freely available textbook
- [Math Cheat Sheets for Data Science](https://github.com/chen115y/DESAL/tree/master/CheatSheets/Math) — quick-reference formula sheets

### Python Programming

Python is the most widely used language in data science and machine learning. These resources will help you get up and running.

- [Python Introduction Notebook](./Python_Introduction/Python_Basics.ipynb) — learn the basics interactively
- [Jupyter Notebook Quick-Start Guide](./Python_Introduction/Quick_Start_Guide.ipynb) — get comfortable with the notebook environment
- [Python Cheat Sheets](https://github.com/chen115y/DESAL/tree/master/CheatSheets/Python) — handy references for everyday Python
- [Python Learning Roadmap](https://aigents.co/learn/roadmaps/python-roadmap) — a structured path from beginner to proficient

---

## Data Science

### Data Science Life Cycle

Every data-science project follows a series of steps — from understanding the problem to deploying a solution. These resources walk you through that process.

- [Data Science Life Cycle — Slide Deck](./Data-Science-Life-Cycle.pdf) — a visual overview of each phase
- [End-to-End Machine Learning Project](./DSLC/02_end_to_end_machine_learning_project.ipynb) — a complete, hands-on walkthrough
- [Principles, Standards & Best Practices](./DSLC/dslc_stardards_best_practices.ipynb) — guidelines for professional-quality work
- [Jupyter Notebook Project Template](./DSLC/template.ipynb) — a reusable starting point for new projects
- [AutoML & Auto-Keras: Getting Started](./DSLC/auto-keras.ipynb) — let the machine choose the best model for you
- [Rules of Machine Learning (Google)](./DSLC/RulesofMachineLearning.pdf) — time-tested best practices for ML engineering
- [MLOps: Continuous Delivery & Automation Pipelines (Google Cloud)](https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — how to operationalize ML models
- [Darts — Time-Series Forecasting Made Easy](https://unit8co.github.io/darts/README.html#forecasting-models) — a Python library for forecasting

### Data Wrangling & Feature Engineering

Raw data is rarely ready for modeling. These tools and tutorials show you how to clean, transform, and prepare your data.

- [NumPy](./DataWrangling/Numpy.ipynb) — fast numerical computing with arrays
- [Pandas](./DataWrangling/Pandas.ipynb) — the go-to library for tabular data manipulation
- [Polars](https://github.com/pola-rs/polars) — a blazing-fast alternative to Pandas
- [PySpark](./DataWrangling/PySpark.ipynb) — process large-scale data with Apache Spark
- [Data Profiling Tools](https://towardsdatascience.com/awesome-data-science-tools-to-master-in-2023-data-profiling-edition-29d29310f779) — quickly understand your dataset's shape and quality
- [Cheat Sheets](https://github.com/chen115y/DESAL/tree/master/CheatSheets/Python)

### Data Visualization

Good visualizations make data understandable at a glance. Learn how to create clear, informative charts and plots.

- [Matplotlib](./Visualization/Matplotlib.ipynb) — the foundational plotting library in Python
- [Seaborn](./Visualization/Seaborn.ipynb) — beautiful statistical visualizations built on Matplotlib
- [Plotly](https://github.com/plotly/plotly.py) — interactive, web-ready charts
- [Visualization Cheat Sheets](https://github.com/chen115y/DESAL/tree/master/CheatSheets/Visualization)

---

## Machine Learning

Machine learning is the practice of teaching computers to learn patterns from data, rather than being explicitly programmed. The resources below cover both classical approaches and modern deep learning.

### Classical Machine Learning

These algorithms form the backbone of most real-world ML systems. They are well-understood, interpretable, and often the right first choice.

- [Categories of ML Algorithms — Visual Map](https://static.coggle.it/diagram/WHeBqDIrJRk-kDDY/t/categories-of-algorithms-non-exhaustive) — see how different algorithms relate to each other
- [Association Rules](./ConventionalMachineLearning/sampleassociation.ipynb) — discover relationships between items (e.g., "customers who buy X also buy Y")
- [Classification](https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb) — predict which category something belongs to
- [Regression](https://github.com/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb) — predict a numerical value
- [Decision Trees](https://github.com/ageron/handson-ml2/blob/master/06_decision_trees.ipynb) — simple, interpretable models based on a series of yes/no questions
- [Ensemble Learning & Random Forests](https://github.com/ageron/handson-ml2/blob/master/07_ensemble_learning_and_random_forests.ipynb) — combine multiple models for better accuracy
- [Unsupervised Learning](https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb) — find hidden structure in data without labels
- [Dimensionality Reduction](https://github.com/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb) — simplify data by reducing the number of features
- [ML Cheat Sheets](https://github.com/chen115y/DESAL/tree/master/CheatSheets/MachineLearning)

### Deep Learning

Deep learning uses neural networks with many layers to learn complex patterns. It powers applications like image recognition, language translation, and generative AI.

- [Neural Networks with Keras](https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb) — build your first neural network
- [Activation Functions Explained](https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/) — understand the math behind how neurons "fire"
- [Training Deep Neural Networks](https://github.com/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb) — techniques for training larger, deeper models
- [Convolutional Neural Networks (CNNs)](https://github.com/ageron/handson-ml2/blob/master/14_deep_computer_vision_with_cnns.ipynb) — the architecture behind computer vision
- [Recurrent Neural Networks (RNNs)](https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb) — models designed for sequential data like text and time series
- [Autoencoders & GANs](https://github.com/ageron/handson-ml2/blob/master/17_autoencoders_and_gans.ipynb) — learn to compress data and generate realistic new samples
- [Deep Learning Cheat Sheets](https://github.com/chen115y/DESAL/tree/master/CheatSheets/DeepLearning)

---

## Natural Language Processing (NLP)

NLP is the branch of AI that helps computers understand, interpret, and generate human language. The Transformer architecture — the foundation of models like GPT and BERT — has revolutionized this field.

- [Word Embeddings — Word2Vec](https://gist.github.com/aparrish/2f562e3737544cf29aaf1af30362f469) — how computers represent words as numbers
- [Advanced NLP with spaCy](https://course.spacy.io/en/) — a free, interactive course on a popular NLP library
- [NLP with RNNs and Attention](https://github.com/ageron/handson-ml2/blob/master/16_nlp_with_rnns_and_attention.ipynb) — sequence models for text
- [Transformer Introduction](./NLP/transformers.ipynb) — the architecture that started the modern AI revolution
- [Transformer for Language Understanding](./NLP/transformer.ipynb) — a hands-on implementation
- [How Transformers Work in NLP](https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/) — a plain-English overview
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) — one of the best visual explanations available
- [A Visual Guide to BERT](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) — understand BERT through diagrams
- [Large Language Models in Practice (Cohere)](https://jalammar.github.io/applying-large-language-models-cohere/) — applying LLMs to real-world tasks
- [How Transformer Attention Works](https://towardsdatascience.com/how-to-explain-transformer-self-attention-like-i-am-five-9020bf50b764) — an intuitive, visual explanation

---

## Generative AI & Large Language Models

Generative AI can create text, images, code, and more. Large language models (LLMs) like GPT, LLaMA, and Claude are at the center of this revolution.

- [How GPT-2 Works](https://blog.floydhub.com/gpt2/) — a deep dive into the model that started the GPT era
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) — visual walkthrough of GPT-2's architecture
- [How GPT-3 Works — Visualizations & Animations](https://jalammar.github.io/how-gpt3-works-visualizations-animations/) — a beautifully illustrated explainer
- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) — how AI generates images from text
- [LMFlow](https://github.com/OptimalScale/LMFlow) — a toolbox for fine-tuning large models
- [LLaMA-Adapter](https://github.com/zrrskywalker/llama-adapter) — efficient fine-tuning for LLaMA models
- [Open-Source Fine-Tuned LLMs](https://medium.com/geekculture/list-of-open-sourced-fine-tuned-large-language-models-llm-8d95a2e0dc76) — a curated directory
- [Google Cloud — Generative AI Learning Path](https://www.cloudskillsboost.google/paths/118) — structured, free courses from Google
- [Lil'Log (Lilian Weng)](https://lilianweng.github.io/) — in-depth technical blog posts on AI research
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook) — practical recipes for working with OpenAI's API
- [I-JEPA](https://ai.facebook.com/blog/yann-lecun-ai-model-i-jepa/) — Meta's step toward more human-like AI understanding
- [(Almost) Everything I Know About LLMs](https://barryz-architecture-of-agentic-llm.notion.site/Almost-Everything-I-know-about-LLMs-d117ca25d4624199be07e9b0ab356a77) — a comprehensive personal knowledge base

### Retrieval-Augmented Generation (RAG)

RAG combines the power of LLMs with external knowledge sources, allowing models to access up-to-date information beyond their training data.

- [RAG Overview — OpenAI](https://openai.com/blog/retrieval-augmented-generation/)
- [RAG Techniques — Curated Collection](https://github.com/NirDiamant/RAG_Techniques)

### Prompt Engineering

Prompt engineering is the art of crafting inputs that get the best results from AI models.

- [Anthropic — Prompt Engineering Overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [Awesome Prompt Engineering](https://github.com/promptslab/Awesome-Prompt-Engineering)

---

## Reinforcement Learning

Reinforcement learning (RL) trains agents to make decisions by rewarding desired behaviors — the same approach used to fine-tune models like ChatGPT through human feedback (RLHF).

- [Reinforcement Learning — An Introduction](https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb) — hands-on notebook
- [Illustrating RLHF](https://huggingface.co/blog/rlhf) — how human feedback shapes AI behavior
- [StackLLaMA: Training LLaMA with RLHF](https://huggingface.co/blog/stackllama) — a practical walkthrough
- [ColossalChat: Cloning ChatGPT with RLHF](https://medium.com/@yangyou_berkeley/colossalchat-an-open-source-solution-for-cloning-chatgpt-with-a-complete-rlhf-pipeline-5edf08fb538b) — an open-source RLHF pipeline
- [RLHF Workshop on AWS](https://www.youtube.com/watch?v=-0pvrCLd2Ak) — hands-on video workshop

---

## Explainability & Data-Centric AI

Understanding *why* a model makes a prediction is just as important as the prediction itself. Data-centric AI shifts focus from tweaking algorithms to improving the quality of training data.

- [Data-Centric AI — MIT Lecture Series](https://www.youtube.com/playlist?app=desktop&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5) — video lectures from MIT
- [Interfaces for Explaining Transformers](https://jalammar.github.io/explaining-transformers/) — visual tools for model interpretation
- [Explainable AI Cheat Sheet](https://jalammar.github.io/explainable-ai/) — a quick visual reference
- [Explainable AI in Practice (LIME & SHAP)](https://www.analyticsvidhya.com/blog/2020/10/unveiling-the-black-box-model-using-explainable-ai-lime-shap-industry-use-case/) — real-world industry use cases
- [Data-Centric AI Concepts Behind GPT](https://towardsdatascience.com/what-are-the-data-centric-ai-concepts-behind-gpt-models-a590071bb727) — how data quality drives model performance

---

## MLOps Platforms

MLOps (Machine Learning Operations) brings DevOps practices to machine learning — making models reproducible, deployable, and maintainable in production.

### Open-Source Tools

| Tool | Description |
|------|-------------|
| [conda](https://github.com/conda/conda) | Package and environment management |
| [Kubernetes](https://kubernetes.io/) | Container orchestration for scalable deployments |
| [Kubeflow](https://www.kubeflow.org/) | ML workflows on Kubernetes |
| [MLflow](https://mlflow.org/) | Experiment tracking, model registry, and deployment |

### Cloud Platforms

| Platform | Description |
|----------|-------------|
| [Databricks MLOps](./The-Big-Book-of-MLOps-v6.pdf) | Unified analytics and AI platform |
| [AWS SageMaker](https://aws.amazon.com/sagemaker/) | End-to-end ML on Amazon Web Services |
| [Google Vertex AI](https://cloud.google.com/vertex-ai) | Google Cloud's ML platform |
| [Azure Machine Learning](https://azure.microsoft.com/en-us/products/machine-learning/) | Microsoft's cloud ML service |

---

## Video Courses & Talks

Sometimes watching is easier than reading. These videos and playlists cover a wide range of topics, from the basics of gradient descent to full deep-learning courses.

**Fundamentals:**
- [Gradient Descent Explained](https://www.youtube.com/watch?v=IHZwWFHWa-w) — the core optimization algorithm behind ML
- [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) — how neural networks learn from their mistakes
- [ML & Deep Learning Fundamentals (Playlist)](https://www.youtube.com/playlist?list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU)

**Deep Learning:**
- [TensorFlow — ML Zero to Hero (25 videos)](https://www.youtube.com/watch?v=KNAWp2S3w94&list=RDCMUC0rqucBdTuFTjJiefW5t-IQ&start_radio=1&t=5)
- [Deep Learning with PyTorch — Full Course (~10 hours)](https://www.youtube.com/watch?v=GIsg-ZUy0MY)

**Neural Network Architectures:**
- [Illustrated Guide to RNNs](https://www.youtube.com/watch?v=LHXXI4-IEns)
- [Illustrated Guide to LSTMs & GRUs](https://www.youtube.com/watch?v=8HyCNIVRbSU)
- [Illustrated Guide to Transformers](https://www.youtube.com/watch?v=4Bdc55j80l8)
- [Transformer Neural Networks Explained](https://www.youtube.com/watch?v=TQQlZhbC5ps)
- [LSTM Is Dead. Long Live Transformers!](https://www.youtube.com/watch?v=S27pHKBEp30)

**Large Language Models:**
- [BERT Explained](https://www.youtube.com/watch?v=xI0HHN5XKDo)
- [GPT-3 Paper Explained](https://www.youtube.com/watch?v=SY5PvZrJhLE)

**Other:**
- [Amazon Machine Learning University (YouTube Channel)](https://www.youtube.com/channel/UC12LqyqTQYbXatYS9AA7Nuw)

---

## Books & Extended Reading

These are freely available textbooks, courses, and reference materials for deeper study.

- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) — comprehensive guide to NumPy, Pandas, Matplotlib, and scikit-learn ([GitHub](https://github.com/jakevdp/PythonDataScienceHandbook))
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow](http://index-of.es/Varios-2/Hands%20on%20Machine%20Learning%20with%20Scikit%20Learn%20and%20Tensorflow.pdf) — one of the most popular ML textbooks ([GitHub](https://github.com/ageron/handson-ml2))
- [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361) — how model size, data, and compute relate to performance
- [Stanford CS324 — Large Language Models](https://stanford-cs324.github.io/winter2022/lectures/introduction/) — understanding and developing LLMs
- [Cornell CS4780 — Machine Learning for Intelligent Systems](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/index.html) — full university course materials
- [An Introduction to the Math Behind Neural Networks](https://hackernoon.com/a-6ur13zzx)
- [Computer Science Courses with Video Lectures](https://github.com/Developer-Y/cs-video-courses) — a massive curated list
- [Data Scientist Interview Questions](./DataScienceInterviewQuestions.pdf)

**Python learning:**
- [Google Python Class](https://developers.google.com/edu/python)
- [Online Python Tutorials](https://pythonspot.com/)
- [Interactive Python (University of Waterloo)](https://cscircles.cemc.uwaterloo.ca/)
- [Algorithms & Data Structures Using Python](https://runestone.academy/runestone/books/published/pythonds/index.html)

**Additional tools & frameworks:**
- [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [MXNet](https://mxnet.apache.org/) — the three major deep-learning frameworks
- [AWS SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples) — notebooks for every SageMaker feature
- [OpenAI Gym — Reinforcement Learning](https://github.com/dennybritz/reinforcement-learning)
- [Spark Data Processing Course](https://github.com/luisbelloch/data_processing_course)
- [PyImageSearch — Computer Vision & OpenCV](https://www.pyimagesearch.com/)

---

## Contributing

Contributions are welcome! If you find a broken link, want to suggest a new resource, or would like to improve an existing notebook, feel free to open an issue or submit a pull request.

---

<p align="center"><em>Happy learning!</em></p>
