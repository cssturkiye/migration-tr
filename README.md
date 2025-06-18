# Migration-TR: Turkish Migration Discourse Dataset 🐦🇹🇷

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange)](https://github.com/cssturkiye/Migration-TR)
[![Dataset](https://img.shields.io/badge/Dataset-Migration--TR-green)](mailto:info@csstr.org)
[![Access](https://img.shields.io/badge/Access-Controlled--See%20DUA-red)](./DATA_USE_AGREEMENT.md)

*Migration Discourses on X.com: Analysis of Public Perceptions and Attitudes Toward Migrants and Refugees in Turkey Using Natural Language Processing*

**Evrim Yılmaz Polat*** and **Evrim Çağın Polat**  
Department of Sociology, Zonguldak Bülent Ecevit University, Zonguldak, Turkey;  
Notrino Research, ODTÜ Teknokent, Ankara, Turkey

*[Computational Social Sciences Turkey (CSSTR)](https://csstr.org) - Computational Social Sciences Working Group*

</div>

---

## 🎯 Overview

This repository contains the **Migration-TR** dataset and accompanying AI models for analyzing migration discourse in Turkish social media. Our research analyzes **6 million tweets** collected between 2011-2022 using the Twitter Academic API, focusing on public perceptions and attitudes toward migrants and refugees in Turkey.

*Full corpus retained internally; external access limited to 500 hydrated tweets per recipient per calendar day, via the DUA.*

### 🔑 Key Highlights
- **5,884,624** raw Turkish tweets → **3,814,679** cleaned tweets
- **12-year temporal analysis** (2011-2022) of migration discourse
- **6-class perception-attitude classification**: sympathy, neutral, antipathy (security/economic/political/other)
- **LoRA-finetuned TurkishBERTweet** model (F1: 0.77 macro)
- **XGBoost bot detection** model (F1: 0.83)

---

## 📊 Dataset: Migration-TR

### 📋 Dataset Overview
| Attribute | Value |
|-----------|-------|
| **Total Tweets** | 5,884,624 (raw) → 3,814,679 (processed) |
| **Time Period** | January 1, 2011 - December 31, 2022 |
| **Language** | Turkish |
| **Data Source** | Twitter Academic API |
| **Processing** | Cleaned, deduplicated, bot-filtered |
| **Annotation** | Manual labeling + AI classification |

### 🗃️ Data Schema
Our dataset includes 26 comprehensive fields:

**Important**: Fields marked ❌ Confidential are *retained only for internal compliance* and **are not distributed**.

<details>
<summary><b>📋 Click to view complete schema</b></summary>

| Field | Type | Description | Availability |
|-------|------|-------------|--------------|
| `created_at` | datetime | Tweet creation timestamp | ✅ Available |
| `tweet_location` | string | Geographic location (if available) | ✅ Available |
| `text` | string | Tweet content (Turkish) | ✅ Available |
| `retweets` | int | Number of retweets | ✅ Available |
| `replies` | int | Number of replies | ✅ Available |
| `likes` | int | Number of likes | ✅ Available |
| `quote_count` | int | Number of quote tweets | ✅ Available |
| `author_id` | string | Anonymized author identifier | ❌ Confidential |
| `username` | string | Author username | ❌ Confidential |
| `name` | string | Author display name | ❌ Confidential |
| `author_pic` | string | Profile picture URL | ❌ Confidential |
| `author_followers` | int | Follower count | ❌ Confidential |
| `author_listed` | int | Listed count | ❌ Confidential |
| `author_following` | int | Following count | ❌ Confidential |
| `author_tweets` | int | Total tweet count | ❌ Confidential |
| `author_protected` | boolean | Protected account status | ❌ Confidential |
| `author_entities` | json | Profile entities | ❌ Confidential |
| `author_description` | string | Profile bio | ❌ Confidential |
| `author_verified` | boolean | Verification status | ❌ Confidential |
| `author_created_at` | datetime | Account creation date | ❌ Confidential |
| `author_withheld` | string | Withheld status | ❌ Confidential |
| `author_location` | string | Author location | ❌ Confidential |
| `is_duplicate` | boolean | Duplicate detection flag | ✅ Available |
| `bot_prob` | float | Bot probability score (0-1) | ✅ Available |
| `is_bot` | boolean | Bot classification | ✅ Available |
| `all_classes_results` | json | AI model predictions | ✅ Available |

</details>

---

## 🤖 AI Models

### 🧠 Perception-Attitude Classification Model
- **Architecture**: LoRA-finetuned TurkishBERTweet
- **Base Model**: `VRLLab/TurkishBERTweet` (900M tweets pre-trained)
- **Task**: 6-class perception-attitude classification
- **Performance**: F1-macro = 0.77
- **Classes**:
  - 😊 `sympathy` - Positive attitudes toward migrants
  - 😐 `neutral` - Neutral/informational content
  - 🛡️ `antipathy-security` - Security threat concerns
  - 💰 `antipathy-economic` - Economic threat concerns
  - 🏛️ `antipathy-political` - Political threat concerns
  - ⚠️ `antipathy-other` - Other negative attitudes

### 🤖 Bot Detection Model
- **Architecture**: XGBoost (ONNX format)
- **Features**: 17 user behavior and profile characteristics
- **Performance**: F1 = 0.83
- **Purpose**: Identify and filter bot-generated content
- **Input**: User profile metadata and behavioral patterns

---

## 🚀 Quick Start

### ⚡ Perception-Attitude Classification
```bash
# Install dependencies
pip install -r requirements.txt

# Run perception-attitude classification
python run_inference.py --text "Mültecilere vatandaşlık verilmesin"
# Output: [{'label': 'antipathy-political', 'score': 0.9992}]
```

### 🤖 Bot Detection
```bash
# Run bot detection (requires user features)
python run_bot_detection.py --features user_features.json
# Output: {'is_bot': True, 'bot_probability': 0.78}
```

---

## 📝 Data Access

### 🔐 Access Requirements

**Who Can Access:**
- **Academic Researchers** at accredited institutions
- **Graduate Students** with supervisor approval
- **Policy Researchers** at recognized organizations
- **Non-commercial use only** - no commercial applications

**Not Permitted:**
- Commercial use or monetization
- Surveillance or tracking applications
- Attempts to re-identify users
- Redistribution of raw tweet text

### 📋 Access Process

#### Step 1: Review Data Use Agreement
Read our comprehensive [Data Use Agreement](./DATA_USE_AGREEMENT.md) carefully.

#### Step 2: Submit Request
Email your signed DUA to: **info@csstr.org**

Include:
- Your institutional affiliation
- Research purpose and methodology
- Specific data requirements (which specific data chunk you need: _From Chunk-1 to Chunk-11770_)
- Supervisor information (for students)

#### Step 3: Approval & Delivery
- We review within 5 business days
- Approved users receive secure download links
- Data delivered as password-protected archives
- **Manual delivery: maximum 500 hydrated objects per recipient per day** (non-automated delivery only)

### ⚠️ Important Disclaimer

**Data Delivery Policy**: Due to X.com (formerly Twitter) Developer Policy requirements, we manually deliver:
- **Maximum 500 hydrated tweets per recipient per day** (non-automated delivery via email/SFTP)
- **Multiple researchers can receive data simultaneously** (500 objects per person per day)
- **Academic use only**
- **No public redistribution** of full tweet text allowed
- **24-hour deletion compliance**: CSSTR monitors X Compliance API and will inform recipients; you must delete or mask affected tweets within 24 hours

**Legal Framework**: This dataset complies with:
- X.com Developer Agreement (current version)
- Turkish data protection laws
- GDPR requirements for research
- Academic research ethics standards

---

## 📖 Citation

If you use Migration-TR in your research, please cite:

**Paper Status**: Currently under peer review. Final citation details will be updated upon acceptance.

```bibtex
@article{yilmazpolat_migration_2025,
  title={Migration Discourses on X.com: Analysis of Public Perceptions and Attitudes Toward Migrants and Refugees in Turkey Using Natural Language Processing},
  author={Yılmaz Polat, Evrim and Çağın Polat, Evrim},
  journal={[Under Review]},
  year={2025},
  note={Dataset available at: https://github.com/cssturkiye/Migration-TR}
}
```

**Interim Citation**: For immediate use, you may cite this repository and dataset as above. We will update with the final journal citation once the peer review process is complete.

---

## 🙏 Acknowledgments

### Base Model Attribution
Our perception-attitude classification model is built upon **TurkishBERTweet** base model, trained by the [VRLLab](https://github.com/ViralLab) team (Najafi & Varol, 2024). We extend our sincere gratitude to the original authors for making their outstanding work available to the research community.

**Model Details:**
- **Base Model**: [`VRLLab/TurkishBERTweet`](https://huggingface.co/VRLLab/TurkishBERTweet)  
- **Pre-training**: 894M Turkish tweets (163M parameters)
- **Our Implementation**: LoRA fine-tuning for 6-class perception-attitude classification
- **Preprocessor**: We include the TurkishBERTweet preprocessor with proper attribution

---

## 🤝 Contact

### 👥 Research Team
**Evrim Yılmaz Polat** - *Principal Investigator & Sociologist*  
Department of Sociology, Zonguldak Bülent Ecevit University, Turkey

**Evrim Çağın Polat** - *Co-Author & Researcher*  
Notrino Research, ODTÜ Teknokent, Ankara, Turkey

📧 Email: info@csstr.org  
🏛️ Organization: [Computational Social Sciences Turkey (CSSTR)](https://csstr.org)

### 🏢 Organization
**Computational Social Sciences Turkey (CSSTR)**  
*Computational Social Sciences Working Group*  
📧 Data Access Requests: info@csstr.org  
📍 Location: Turkey

### 💬 Support
- **Data Access**: info@csstr.org
- **Technical Issues**: GitHub Issues
- **Research Collaboration**: info@csstr.org

---

## 📚 References

Najafi, A., & Varol, O. (2024). Turkishbertweet: Fast and reliable large language model for social media analysis. *Expert Systems with Applications*, *255*, 124737.

---

<div align="center">

**Migration-TR** | *Advancing Migration Research Through Computational Social Science*

*Made with ❤️ for the research community*

---

*X is a trademark of X Corp.; use here is for identification only.*

</div>