# Deploy CineSense - step by step

Two things must be published: the **GitHub repo** (your submission link) and the
**Hugging Face Space** (the live deployment URL). Both need to run on your
machine because they use your own GitHub/Hugging Face login - that part can't be
automated for you safely.

Everything below is copy-paste. The project folder is:
`C:\Users\Besfo\Desktop\Schule\Relevant\KI-Anwendungen\cinesense`

---

## 1. Publish to GitHub (~2 min)

Open a terminal in the `cinesense` folder, then:

### Easiest (with GitHub CLI `gh`)
```bash
deploy\publish_github.bat YOUR_GITHUB_USERNAME      # Windows
# or:  ./deploy/publish_github.sh YOUR_GITHUB_USERNAME   # Mac/Linux/Git-Bash
```
This creates the public repo, pushes everything, and invites `jasminh` and `bkuehnis`.

### Without `gh` (plain git)
1. Create an empty repo named `cinesense` at https://github.com/new (no README).
2. In the folder:
   ```bash
   git init
   git add .
   git commit -m "CineSense: ML Numeric + NLP movie rating fusion project"
   git branch -M main
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/cinesense.git
   git push -u origin main
   ```
3. On GitHub: **Settings -> Collaborators -> Add people** -> add **`jasminh`** and **`bkuehnis`**.

Copy the repo URL (e.g. `https://github.com/YOU/cinesense`).

---

## 2. Deploy the Hugging Face Space (~3 min)

```bash
pip install huggingface_hub
huggingface-cli login
# paste a WRITE token from https://huggingface.co/settings/tokens
python deploy/deploy_hf.py YOUR_HF_USERNAME
```
It creates the Gradio Space `YOUR_HF_USERNAME/cinesense`, uploads everything, and
prints the URL: `https://huggingface.co/spaces/YOUR_HF_USERNAME/cinesense`.
Wait ~1 minute for it to build, then open the **App** tab.

**Prefer the website instead?** Go to https://huggingface.co/new-space -> SDK
"Gradio" -> name it `cinesense` -> Create -> "Files" -> "Add file" -> "Upload
files" and drag in `app.py`, `requirements.txt`, the `src/` folder and the
`models/` folder (and optionally the rest).

---

## 3. Finalise the documentation (1 min)

Open `docs/documentation.md` and fill the two placeholders, then tick the boxes:
- **Project Metadata** -> GitHub repository URL + Deployment URL
- **Mandatory Setup Checks** -> tick "Deployment URL provided" and "Required GitHub users added"
- **Section 3. Deployment** -> paste the Space URL, and replace the provisional
  image `reports/figures/13_app_preview.png` with a real screenshot of your
  running Space (open it, press PrtSc / use Snipping Tool, save over that file).

Then commit the update:
```bash
git add docs/documentation.md reports/figures/13_app_preview.png
git commit -m "Add live URLs and deployment screenshot"
git push
```

Done - submit the GitHub link before **2026-06-07 18:00**.
