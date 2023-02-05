# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python38_render_template]
# [START gae_python3_render_template]
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "qf.settings")
os.environ['DJANGO_SETTINGS_MODULE'] = 'qf.settings'

from haystack.telemetry import disable_telemetry
disable_telemetry()

from flask import Flask, render_template
app = Flask(__name__)

from tqdm.auto import tqdm
from haystack.nodes import QuestionGenerator, BM25Retriever, FARMReader
from haystack.pipelines import (
    QuestionGenerationPipeline,
    RetrieverQuestionGenerationPipeline,
    QuestionAnswerGenerationPipeline,
)
from haystack.utils import  print_questions
from haystack.document_stores import ElasticsearchDocumentStore

document_store = ElasticsearchDocumentStore(
    scheme="https",
    port=443,
    host="9613a53212b2417cb704c4c26a8e6c80.us-central1.gcp.cloud.es.io",
    api_key="7Q5YcgPzSm6-rsnqfFmr6A", 
    api_key_id="1nlaAoYBNcW5EzzUl9xP",
    index="search-ai2",
    embedding_field="question_emb",
    embedding_dim=384,
    excluded_meta_data=["question_emb"],
    similarity="cosine",
)


@app.route('/')
def root():
    # Initialize document store and write in the documents
    document_store.delete_documents()
    text1 = "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace."
    text2 = "Princess Arya Stark is the third child and second daughter of Lord Eddard Stark and his wife, Lady Catelyn Stark. She is the sister of the incumbent Westerosi monarchs, Sansa, Queen in the North, and Brandon, King of the Andals and the First Men. After narrowly escaping the persecution of House Stark by House Lannister, Arya is trained as a Faceless Man at the House of Black and White in Braavos, using her abilities to avenge her family. Upon her return to Westeros, she exacts retribution for the Red Wedding by exterminating the Frey male line."
    text3 = "Java is a high-level, class-based, object-oriented programming language that is designed to have as few implementation dependencies as possible. It is a general-purpose programming language intended to let programmers write once, run anywhere (WORA),[17] meaning that compiled Java code can run on all platforms that support Java without the need to recompile.[18] Java applications are typically compiled to bytecode that can run on any Java virtual machine (JVM) regardless of the underlying computer architecture. The syntax of Java is similar to C and C++, but has fewer low-level facilities than either of them. The Java runtime provides dynamic capabilities (such as reflection and runtime code modification) that are typically not available in traditional compiled languages. As of 2019, Java was one of the most popular programming languages in use according to GitHub,[citation not found][19][20] particularly for client–server web applications, with a reported 9 million developers.[21] Java was originally developed by James Gosling at Sun Microsystems. It was released in May 1995 as a core component of Sun Microsystems' Java platform. The original and reference implementation Java compilers, virtual machines, and class libraries were originally released by Sun under proprietary licenses. As of May 2007, in compliance with the specifications of the Java Community Process, Sun had relicensed most of its Java technologies under the GPL-2.0-only license. Oracle offers its own HotSpot Java Virtual Machine, however the official reference implementation is the OpenJDK JVM which is free open-source software and used by most developers and is the default JVM for almost all Linux distributions."

    docs = [{"content": text3}]

    document_store.write_documents(docs)

    # Initialize Question Generator
    question_generator = QuestionGenerator()

    reader = FARMReader("deepset/minilm-uncased-squad2")
    qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)
    for idx, document in enumerate(tqdm(document_store)):
        if(idx != 0): 
            break
        print(f"\n * Generating questions and answers for document {idx}: {document.content[:100]}...\n")
        result = qag_pipeline.run(documents=[document])
        print_questions(result)
            

    return render_template('index.html', times=result["queries"])


# Starts the api
if __name__ == '__main__':
    host = '127.0.0.1'
    port = 8080
    app.run(host=host, port=port, debug=True)