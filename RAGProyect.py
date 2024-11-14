import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

web_paths = [
    "https://medium.com/@brianjking/2017-chicago-mechanical-keyboards-meetup-8e95b687922b",
    "https://medium.com/@joshcamson/magic-keyboard-with-touch-id-a-daily-timesaver-eef14a4161c4",
    "https://medium.com/illumination/logitech-mx-mechanical-keyboard-review-55e2a9441bb7",
    "https://medium.com/@luisdamed/making-a-macro-pad-from-scratch-8b7d1884f2ba",
    "https://medium.com/macoclock/defying-apples-product-designers-replacing-my-magic-keyboard-with-a-mechanical-one-5bc9fa2608e6",
    "https://medium.com/macoclock/keychron-k2-keyboard-review-3e3428e6be87",
    "https://medium.com/@shanjidaaktarshima/the-importance-of-an-seo-audit-report-d494623ce1dc",
    "https://medium.com/tech-travel-and-photography-for-seniors/do-you-love-typing-waitll-you-try-this-4e3fe6fefbf9",
    "https://medium.com/@myam/the-nuphy-air75-v1-is-better-with-karabiner-elements-17ed69df5987",
    "https://switchandclick.com/mechanical-keyboard-switch-guide/",
    "https://switchandclick.com/recommended-tools/",
    "https://switchandclick.com/double-shot-vs-dye-sub-keycaps-whats-the-difference/",
    "https://switchandclick.com/the-best-artisan-keycaps-and-where-to-find-them/",
    "https://switchandclick.com/sa-vs-dsa-vs-oem-vs-cherry-vs-xda-keycap-profiles/",
    "https://switchandclick.com/will-it-fit-finding-keycaps-that-will-fit-your-keyboard/",
    "https://switchandclick.com/gmk-keycaps/",
    "https://switchandclick.com/abs-vs-pbt-keycaps-whats-the-difference/",
    "https://switchandclick.com/best-keycaps/",
    "https://switchandclick.com/how-to-build-a-keyboard/",
    "https://switchandclick.com/how-to-mod-your-stabilizers-band-aid-clip-and-lube/",
    "https://switchandclick.com/5-easy-modifications-to-improve-your-mechanical-keyboard/",
    "https://switchandclick.com/what-lube-to-use-for-mechanical-keyboard-switches/",
    "https://switchandclick.com/stabilizer-guide/",
    "https://switchandclick.com/recommended-tools/"

]

# Cargar y procesar los documentos
loader = WebBaseLoader(
    web_paths=web_paths,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("yoast-schema-graph", "post-template-default", "single", "single-post", "postid-596", "single-format-standard", "aawp-custom", "lasso-v327", "site", "skip-link", "screen-reader-text", "site-header", "desktop-toggle", "toggled", "inner-wrap", "hamburger-wrapper", "desktop", "hamburger", "hamburger--squeeze", "menu-toggle", "hamburger-box", "hamburger-inner", "label", "site-branding", "custom-logo-link", "attachment-medium", "size-medium", "search-wrapper", "sr-only", "search-form", "search-field", "main-navigation", "menu-primary-container", "nav-menu", "menu-item", "menu-item-type-post_type", "menu-item-object-page", "menu-item-home", "menu-item-13212", "menu-item-13213", "menu-item-type-taxonomy", "menu-item-object-category", "current-post-ancestor", "current-menu-parent", "current-post-parent", "menu-item-has-children", "menu-item-29798", "toggle-submenu", "submenu", "menu-item-13231", "menu-item-13232", "menu-item-13233", "menu-item-13234", "menu-item-13235", "menu-item-29799", "menu-item-13240", "menu-item-13241", "menu-item-29800", "menu-item-28962", "menu-item-28963", "menu-item-28964", "menu-item-13244", "site-content", "content-area", "site-main", "post-596", "post", "type-post", "status-publish", "format-standard", "has-post-thumbnail", "hentry", "category-keyboard-mods", "category-keyboard-switch", "category-keyboards", "tag-lubing-switches", "tag-mechanical-keyboard", "tag-stabilizer-lube", "mv-content-wrapper", "entry-header", "entry-title", "entry-meta", "byline", "author", "vcard", "url", "fn", "n", "entry-content", "wp-block-image", "size-large", "sp-no-webp", "wp-block-table", "is-style-regular", "wp-block-heading", "wp-image-1754", "wp-element-caption", "is-resized", "wp-image-1756", "wp-block-embed", "is-type-video", "is-provider-youtube", "wp-block-embed-youtube", "wp-embed-aspect-16-9", "wp-has-aspect-ratio", "wp-block-embed__wrapper", "wp-image-1762", "wp-image-599", "wp-image-1763", "wp-image-1764", "entry-footer", "author-card", "avatar-container", "author-info", "h3", "h2", "recent-articles-container", "article-card", "image-container", "copy-container", "excerpt", "widget-area", "about-wrapper", "widget-title", "about-copy", "legal-info-container", "widget_text", "widget", "widget_custom_html", "textwidget", "custom-html-widget", "editor_picks", "editor-label", "editor-picks", "site-footer", "menu-footer-container", "menu-item-privacy-policy", "menu-item-15983", "menu-item-15984", "menu-item-15985", "menu-item-33257", "site-info", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "du", "ag", "dv", "bf", "ak", "am", "an", "ao", "ap", "aq", "ar", "as", "at", "dw", "dt", "ab", "dx", "dy", "dz", "ea", "eb", "ec", "ed", "ee", "ef", "eg", "eh", "ei", "ej", "ek", "el", "em", "eo", "ep", "eq", "er", "es", "et", "eu", "ev", "ew", "ex", "ey", "ez", "fa", "fb", "fc", "fd", "bm", "fe", "ff", "ax", "af", "ah", "ai", "aj", "al", "ac", "ae", "au", "av", "aw", "ay", "az", "ba", "bb", "bc", "bd", "bn", "bo", "be", "bg", "bh", "bi", "bj", "bk", "bl", "fg", "fh", "fi", "fj", "fk", "fl", "fm", "fn", "fo", "fp", "fq", "by", "bz", "ca", "cx", "fr", "fs", "ft", "ok", "rx", "ry", "rz", "sa", "sb", "sc", "gm", "sd", "se", "sf", "sg", "sh", "si", "sj", "sk", "le", "sl", "sm", "sn", "so", "sp", "sq", "sr", "ss", "st", "fu", "fv", "fw", "fx", "fy", "cb", "ci", "fz", "ga", "gb", "gc", "meteredContent", "gi", "gj", "gk", "gl", "speechify-ignore", "gn", "go", "gp", "gq", "gr", "cl", "gs", "gv", "gt", "gu", "gw", "gx", "gy", "gz", "ha", "pw-post-title", "hb", "hc", "hd", "he", "hf", "hg", "hh", "hi", "hj", "hk", "hl", "hm", "hn", "ho", "hp", "hq", "hr", "hs", "ht", "hu", "hv", "hw", "hx", "hy", "hz", "cp", "ia", "ib", "ic", "id", "ie", "if", "ig", "ih", "ii", "ij", "dd", "de", "ik", "il", "im", "in", "io", "ip", "iq", "ir", "is", "it", "iu", "iv", "iw", "ix", "cn", "iy", "iz", "ja", "jb", "jc", "jd", "je", "jf", "jg", "jh", "ji", "jj", "jk", "jl", "jm", "jn", "jo", "jp", "jq", "jr", "js", "ki", "kj", "kk", "pw-multi-vote-icon", "kl", "km", "kn", "ko", "kp", "kq", "kr", "ks", "kt", "ku", "kv", "pw-multi-vote-count", "kw", "kx", "ky", "kz", "la", "lb", "lc", "su", "lj", "op", "oq", "lf", "lg", "lh", "jt", "ju", "jv", "jw", "jx", "jy", "jz", "ka", "kb", "kc", "kd", "ke", "kf", "kg", "kh", "li", "lk", "ll", "lm", "ln", "lo", "lp", "lq", "lr", "ls", "lt", "lu", "lv", "lw", "lx", "ly", "lz", "ma", "mb", "mc", "md", "me", "mf", "mg", "mh", "mi", "pw-post-body-paragraph", "mj", "mk", "ml", "mm", "mn", "mo", "mp", "mq", "mr", "ms", "mt", "mu", "mv", "mw", "mx", "my", "mz", "na", "nb", "nc", "nd", "ne", "nf", "ng", "nh", "nl", "nm", "nn", "no", "np", "nq", "ni", "nj", "paragraph-image", "nr", "ns", "nt", "nu", "nk", "nv", "nw", "nx", "ny", "nz", "sx", "sy", "sz", "ta", "tb", "tc", "co", "td", "te", "tf", "tg", "th", "ti", "qg", "tj", "qj", "tk", "tl", "tm", "tn", "to", "tp", "tq", "tr", "ts", "tt", "tu", "tv", "tw", "tx", "pp", "ty", "tz", "ua", "ub", "uc", "ud", "gd", "ue", "pu", "uf", "ug", "uh", "ui", "uj", "br", "uk", "ul", "oa", "ob", "oc", "od", "oe", "of", "og", "oh", "oi", "oj", "ol", "om", "on", "oo", "bq", "or", "os", "ot", "ou", "ov", "ow", "bx", "rc", "rd", "re", "rf", "rg", "rh", "sw", "rj", "rk", "pw-author-name", "qf", "ru", "rv", "rw", "qs", "pw-follower-count", "qu", "qv", "qw", "um", "un", "uo", "up", "uq", "ur", "us", "ut", "uu", "uv", "uw", "ux", "uy", "uz", "va", "vb", "vc", "vd", "ve", "vf", "vg", "qd", "vh", "vi", "vj", "qc", "vk", "vl", "vm", "vn", "vo", "vp", "vq", "vr", "vs", "vt", "vu", "vv", "vw", "vx", "vy", "vz", "wa", "wb", "wc", "wd", "we", "wf", "wg", "wh", "wi", "wj", "wk", "wl", "wm", "wn", "wo", "wp", "wr", "ws", "wt", "wu", "wv", "ww", "wx", "wq", "wy", "wz", "xa", "xb", "xc", "xd", "xe", "xf", "xg", "xh", "xi", "xj", "xk", "qe", "xl", "xm", "xn", "xo", "xp", "xq", "xr", "xs", "xt", "xu", "xv", "xw", "xx", "xy", "xz", "ya", "yb", "yc", "yd", "ye", "yf", "yg", "yh", "yi", "yj", "yk", "yl", "ym", "yn", "yo", "yp", "yq", "qm", "yr", "ys", "qn", "qo", "yt", "yu", "qp", "qq", "yv", "yw", "qr", "yx", "yy", "yz", "zc", "za", "dh", "zb", "dj", "zd", "ze", "zf", "zg", "dk", "dl", "zh", "zi", "zj", "oz", "oy", "ox", "zk", "zl", "zm", "zn", "zo", "zp", "zq", "zr", "zs", "zt", "zu", "zv", "zw", "zx", "zy", "zz", "aba", "abb", "abc", "abd", "abe", "abf", "abg", "abh", "abi", "abj", "abp", "abq", "abl", "abm", "abn", "abo", "abk", "abr", "abs", "abt", "abu", "abv", "abw", "abx", "aby", "abz", "aca", "qy", "qz", "ra", "rb", "grecaptcha-badge", "grecaptcha-logo", "grecaptcha-error", "g-recaptcha-response"
)
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_rag_response(question: str) -> str:
    return rag_chain.invoke(question)