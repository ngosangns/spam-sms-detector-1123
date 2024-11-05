import ipaddress
import os
import pickle
import re
import socket
import urllib.request
from datetime import date
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import whois
from bs4 import BeautifulSoup
from catboost import CatBoostClassifier as CatBoost
from googlesearch import search
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import (
    GradientBoostingClassifier as SklearnGradientBoostingClassifier,
)
from sklearn.ensemble import (
    RandomForestClassifier as SklearnRandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as SKLearnDecisionTree


class FeatureExtraction:
    features = []

    def __init__(self, url):
        self.features = []
        self.url = url
        self.domain = ""
        self.whois_response = ""
        self.urlparse = ""
        self.response = ""
        self.soup = ""

        try:
            self.response = requests.get(url)
            self.soup = BeautifulSoup(self.response.text, "html.parser")
        except Exception:
            pass

        try:
            self.urlparse = urlparse(url)
            self.domain = self.urlparse.netloc
        except Exception:
            pass

        try:
            self.whois_response = whois.whois(self.domain)
        except Exception:
            pass

        self.features.append(self.UsingIp())
        self.features.append(self.longUrl())
        self.features.append(self.shortUrl())
        self.features.append(self.symbol())
        self.features.append(self.redirecting())
        self.features.append(self.prefixSuffix())
        self.features.append(self.SubDomains())
        self.features.append(self.Hppts())
        self.features.append(self.DomainRegLen())
        self.features.append(self.Favicon())

        self.features.append(self.NonStdPort())
        self.features.append(self.HTTPSDomainURL())
        self.features.append(self.RequestURL())
        self.features.append(self.AnchorURL())
        self.features.append(self.LinksInScriptTags())
        self.features.append(self.ServerFormHandler())
        self.features.append(self.InfoEmail())
        self.features.append(self.AbnormalURL())
        self.features.append(self.WebsiteForwarding())
        self.features.append(self.StatusBarCust())

        self.features.append(self.DisableRightClick())
        self.features.append(self.UsingPopupWindow())
        self.features.append(self.IframeRedirection())
        self.features.append(self.AgeofDomain())
        self.features.append(self.DNSRecording())
        self.features.append(self.WebsiteTraffic())
        self.features.append(self.PageRank())
        self.features.append(self.GoogleIndex())
        self.features.append(self.LinksPointingToPage())
        self.features.append(self.StatsReport())

    # 1.UsingIp
    def UsingIp(self):
        try:
            ipaddress.ip_address(self.url)
            return -1
        except Exception:
            return 1

    # 2.longUrl
    def longUrl(self):
        if len(self.url) < 54:
            return 1
        if len(self.url) >= 54 and len(self.url) <= 75:
            return 0
        return -1

    # 3.shortUrl
    def shortUrl(self):
        match = re.search(
            "bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
            "yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
            "short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
            "doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|"
            "db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|"
            "q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|"
            "x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net",
            self.url,
        )
        if match:
            return -1
        return 1

    # 4.Symbol@
    def symbol(self):
        if re.findall("@", self.url):
            return -1
        return 1

    # 5.Redirecting//
    def redirecting(self):
        if self.url.rfind("//") > 6:
            return -1
        return 1

    # 6.prefixSuffix
    def prefixSuffix(self):
        try:
            match = re.findall("\-", self.domain)
            if match:
                return -1
            return 1
        except Exception:
            return -1

    # 7.SubDomains
    def SubDomains(self):
        dot_count = len(re.findall("\.", self.url))
        if dot_count == 1:
            return 1
        elif dot_count == 2:
            return 0
        return -1

    # 8.HTTPS
    def Hppts(self):
        try:
            https = self.urlparse.scheme
            if "https" in https:
                return 1
            return -1
        except Exception:
            return 1

    # 9.DomainRegLen
    def DomainRegLen(self):
        try:
            expiration_date = self.whois_response.expiration_date
            creation_date = self.whois_response.creation_date
            try:
                if len(expiration_date):
                    expiration_date = expiration_date[0]
            except Exception:
                pass
            try:
                if len(creation_date):
                    creation_date = creation_date[0]
            except Exception:
                pass

            age = (expiration_date.year - creation_date.year) * 12 + (
                expiration_date.month - creation_date.month
            )
            if age >= 12:
                return 1
            return -1
        except Exception:
            return -1

    # 10. Favicon
    def Favicon(self):
        try:
            for head in self.soup.find_all("head"):
                for head.link in self.soup.find_all("link", href=True):
                    dots = [x.start(0) for x in re.finditer("\.", head.link["href"])]
                    if (
                        self.url in head.link["href"]
                        or len(dots) == 1
                        or self.domain in head.link["href"]
                    ):
                        return 1
            return -1
        except Exception:
            return -1

    # 11. NonStdPort
    def NonStdPort(self):
        try:
            port = self.domain.split(":")
            if len(port) > 1:
                return -1
            return 1
        except Exception:
            return -1

    # 12. HTTPSDomainURL
    def HTTPSDomainURL(self):
        try:
            if "https" in self.domain:
                return -1
            return 1
        except Exception:
            return -1

    # 13. RequestURL
    def RequestURL(self):
        try:
            i, success = 0, 0
            for img in self.soup.find_all("img", src=True):
                dots = [x.start(0) for x in re.finditer("\.", img["src"])]
                if (
                    self.url in img["src"]
                    or self.domain in img["src"]
                    or len(dots) == 1
                ):
                    success = success + 1
                i = i + 1

            for audio in self.soup.find_all("audio", src=True):
                dots = [x.start(0) for x in re.finditer("\.", audio["src"])]
                if (
                    self.url in audio["src"]
                    or self.domain in audio["src"]
                    or len(dots) == 1
                ):
                    success = success + 1
                i = i + 1

            for embed in self.soup.find_all("embed", src=True):
                dots = [x.start(0) for x in re.finditer("\.", embed["src"])]
                if (
                    self.url in embed["src"]
                    or self.domain in embed["src"]
                    or len(dots) == 1
                ):
                    success = success + 1
                i = i + 1

            for iframe in self.soup.find_all("iframe", src=True):
                dots = [x.start(0) for x in re.finditer("\.", iframe["src"])]
                if (
                    self.url in iframe["src"]
                    or self.domain in iframe["src"]
                    or len(dots) == 1
                ):
                    success = success + 1
                i = i + 1

            try:
                percentage = success / float(i) * 100
                if percentage < 22.0:
                    return 1
                elif (percentage >= 22.0) and (percentage < 61.0):
                    return 0
                else:
                    return -1
            except Exception:
                return 0
        except Exception:
            return -1

    # 14. AnchorURL
    def AnchorURL(self):
        try:
            i, unsafe = 0, 0
            for a in self.soup.find_all("a", href=True):
                if (
                    "#" in a["href"]
                    or "javascript" in a["href"].lower()
                    or "mailto" in a["href"].lower()
                    or not (self.url in a["href"] or self.domain in a["href"])
                ):
                    unsafe = unsafe + 1
                i = i + 1

            try:
                percentage = unsafe / float(i) * 100
                if percentage < 31.0:
                    return 1
                elif (percentage >= 31.0) and (percentage < 67.0):
                    return 0
                else:
                    return -1
            except Exception:
                return -1

        except Exception:
            return -1

    # 15. LinksInScriptTags
    def LinksInScriptTags(self):
        try:
            i, success = 0, 0

            for link in self.soup.find_all("link", href=True):
                dots = [x.start(0) for x in re.finditer("\.", link["href"])]
                if (
                    self.url in link["href"]
                    or self.domain in link["href"]
                    or len(dots) == 1
                ):
                    success = success + 1
                i = i + 1

            for script in self.soup.find_all("script", src=True):
                dots = [x.start(0) for x in re.finditer("\.", script["src"])]
                if (
                    self.url in script["src"]
                    or self.domain in script["src"]
                    or len(dots) == 1
                ):
                    success = success + 1
                i = i + 1

            try:
                percentage = success / float(i) * 100
                if percentage < 17.0:
                    return 1
                elif (percentage >= 17.0) and (percentage < 81.0):
                    return 0
                else:
                    return -1
            except Exception:
                return 0
        except Exception:
            return -1

    # 16. ServerFormHandler
    def ServerFormHandler(self):
        try:
            if len(self.soup.find_all("form", action=True)) == 0:
                return 1
            else:
                for form in self.soup.find_all("form", action=True):
                    if form["action"] == "" or form["action"] == "about:blank":
                        return -1
                    elif (
                        self.url not in form["action"]
                        and self.domain not in form["action"]
                    ):
                        return 0
                    else:
                        return 1
        except Exception:
            return -1

    # 17. InfoEmail
    def InfoEmail(self):
        try:
            if re.findall(r"[mail\(\)|mailto:?]", self.soap):
                return -1
            else:
                return 1
        except Exception:
            return -1

    # 18. AbnormalURL
    def AbnormalURL(self):
        try:
            if self.response.text == self.whois_response:
                return 1
            else:
                return -1
        except Exception:
            return -1

    # 19. WebsiteForwarding
    def WebsiteForwarding(self):
        try:
            if len(self.response.history) <= 1:
                return 1
            elif len(self.response.history) <= 4:
                return 0
            else:
                return -1
        except Exception:
            return -1

    # 20. StatusBarCust
    def StatusBarCust(self):
        try:
            if re.findall("<script>.+onmouseover.+</script>", self.response.text):
                return 1
            else:
                return -1
        except Exception:
            return -1

    # 21. DisableRightClick
    def DisableRightClick(self):
        try:
            if re.findall(r"event.button ?== ?2", self.response.text):
                return 1
            else:
                return -1
        except Exception:
            return -1

    # 22. UsingPopupWindow
    def UsingPopupWindow(self):
        try:
            if re.findall(r"alert\(", self.response.text):
                return 1
            else:
                return -1
        except Exception:
            return -1

    # 23. IframeRedirection
    def IframeRedirection(self):
        try:
            if re.findall(r"[<iframe>|<frameBorder>]", self.response.text):
                return 1
            else:
                return -1
        except Exception:
            return -1

    # 24. AgeofDomain
    def AgeofDomain(self):
        try:
            creation_date = self.whois_response.creation_date
            try:
                if len(creation_date):
                    creation_date = creation_date[0]
            except Exception:
                pass

            today = date.today()
            age = (today.year - creation_date.year) * 12 + (
                today.month - creation_date.month
            )
            if age >= 6:
                return 1
            return -1
        except Exception:
            return -1

    # 25. DNSRecording
    def DNSRecording(self):
        try:
            creation_date = self.whois_response.creation_date
            try:
                if len(creation_date):
                    creation_date = creation_date[0]
            except Exception:
                pass

            today = date.today()
            age = (today.year - creation_date.year) * 12 + (
                today.month - creation_date.month
            )
            if age >= 6:
                return 1
            return -1
        except Exception:
            return -1

    # 26. WebsiteTraffic
    def WebsiteTraffic(self):
        try:
            rank = BeautifulSoup(
                urllib.request.urlopen(
                    "http://data.alexa.com/data?cli=10&dat=s&url=" + self.url
                ).read(),
                "xml",
            ).find("REACH")["RANK"]
            if int(rank) < 100000:
                return 1
            return 0
        except Exception:
            return -1

    # 27. PageRank
    def PageRank(self):
        try:
            rank_checker_response = requests.post(
                "https://www.checkpagerank.net/index.php", {"name": self.domain}
            )

            global_rank = int(
                re.findall(r"Global Rank: ([0-9]+)", rank_checker_response.text)[0]
            )
            if global_rank > 0 and global_rank < 100000:
                return 1
            return -1
        except Exception:
            return -1

    # 28. GoogleIndex
    def GoogleIndex(self):
        try:
            site = search(self.url, 5)
            if site:
                return 1
            else:
                return -1
        except Exception:
            return 1

    # 29. LinksPointingToPage
    def LinksPointingToPage(self):
        try:
            number_of_links = len(re.findall(r"<a href=", self.response.text))
            if number_of_links == 0:
                return 1
            elif number_of_links <= 2:
                return 0
            else:
                return -1
        except Exception:
            return -1

    # 30. StatsReport
    def StatsReport(self):
        try:
            url_match = re.search(
                "at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly",
                self.url,
            )
            ip_address = socket.gethostbyname(self.domain)
            ip_match = re.search(
                "146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|"
                "107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|"
                "118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|"
                "216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|"
                "34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|"
                "216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42",
                ip_address,
            )
            if url_match:
                return -1
            elif ip_match:
                return -1
            return 1
        except Exception:
            return 1

    def getFeaturesList(self):
        return self.features


class URLClassifier:
    """
    Lớp URLClassifier dùng để phân loại tin nhắn URL thành spam hoặc không spam.
    Attributes:
        model_name (str): Tên của mô hình.
        model_dir (str): Thư mục lưu trữ kết quả.
        model_path (str): Đường dẫn đến file lưu trữ mô hình.
        model (sklearn model): Mô hình học máy dùng để phân loại.
    Methods:
        __init__(model_name, model_dir):
            Khởi tạo đối tượng URLClassifier với tên mô hình và thư mục kết quả.
        load_data(directory):
            Tải dữ liệu URL từ file CSV chỉ định và trả về danh sách tin nhắn và nhãn tương ứng.
        balance_dataset(X_train, y_train):
            Cân bằng tập dữ liệu huấn luyện bằng cách sử dụng kỹ thuật oversampling.
        preprocess_data(url_data, labels):
            Tiền xử lý dữ liệu URL và nhãn, bao gồm chia tập dữ liệu.
        train_model(X_train, y_train):
            Huấn luyện mô hình với dữ liệu huấn luyện.
        evaluate_model(X_test, y_test):
            Đánh giá mô hình với dữ liệu kiểm tra và trả về nhãn thực tế và nhãn dự đoán.
        save_model():
            Lưu mô hình đã huấn luyện vào file.
        load_model():
            Tải mô hình từ file.
    """

    def __init__(self, model_name, model_dir):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_path = os.path.join(
            self.model_dir, f"url-{self.model_name}-model.pkl"
        )
        self.model = None

    def load_data(self, csv_path):
        data = pd.read_csv(csv_path)
        data = data.drop(["Index"], axis=1)  # Drop the Index column

        url_data = data.drop(["class"], axis=1)
        label = data["class"]

        return url_data, label

    def balance_dataset(self, X_train, y_train):
        oversampler = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    def preprocess_data(self, url_data, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            url_data, labels, test_size=0.2, random_state=42
        )

        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        # print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
        # print(classification_report(y_test, y_pred))

        return y_test, y_pred

    def predict(self, url):
        obj = FeatureExtraction(url)
        extracted_features = np.array(obj.getFeaturesList()).reshape(1, 30)

        # y_pro_phishing = self.model.predict_proba(obj)[0, 0]
        # y_pro_non_phishing = self.model.predict_proba(obj)[0, 1]

        return self.model.predict(extracted_features)[0]  # 1 is safe, -1 is unsafe

    def save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump((self.model), f)

    def load_model(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)


class SVMClassifier(URLClassifier):
    def __init__(self, model_dir):
        super().__init__("svm", model_dir)

        # defining parameter range
        param_grid = {"gamma": [0.1], "kernel": ["rbf", "linear"]}

        self.model = GridSearchCV(SVC(), param_grid)


class NaiveBayesClassifier(URLClassifier):
    def __init__(self, model_dir):
        super().__init__("naive-bayes", model_dir)
        self.model = GaussianNB()


class RandomForestClassifier(URLClassifier):
    def __init__(self, model_dir):
        super().__init__("random-forest", model_dir)
        self.model = SklearnRandomForestClassifier(n_estimators=10)


class LogisticRegressionClassifier(URLClassifier):
    def __init__(self, model_dir):
        super().__init__("logistic-regression", model_dir)
        self.model = LogisticRegression()


class KNNClassifier(URLClassifier):
    def __init__(self, model_dir):
        super().__init__("knn", model_dir)
        self.model = KNeighborsClassifier(n_neighbors=1)


class GradientBoostingClassifier(URLClassifier):
    def __init__(self, model_dir):
        super().__init__("gradient-boosting", model_dir)
        self.model = SklearnGradientBoostingClassifier(max_depth=4, learning_rate=0.7)


class DecisionTreeClassifier(URLClassifier):
    def __init__(self, model_dir):
        super().__init__("decision-tree", model_dir)
        self.model = SKLearnDecisionTree(max_depth=30)


class CatBoostClassifier(URLClassifier):
    def __init__(self, model_dir):
        super().__init__("catboost", model_dir)
        self.model = CatBoost(learning_rate=0.1)


class MLPClassifier(URLClassifier):
    def __init__(self, model_dir):
        super().__init__("mlp", model_dir)
        self.model = MLP()
