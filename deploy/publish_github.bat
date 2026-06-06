@echo off
REM Publish CineSense to GitHub. Usage: deploy\publish_github.bat YOUR_GITHUB_USERNAME [repo_name]
if "%~1"=="" (echo Usage: deploy\publish_github.bat YOUR_GITHUB_USERNAME [repo_name] & exit /b 1)
set USER=%~1
set REPO=%~2
if "%REPO%"=="" set REPO=cinesense
cd /d "%~dp0.."
git init
git add .
git commit -m "CineSense: ML Numeric + NLP movie rating fusion project"
git branch -M main
where gh >nul 2>nul
if %errorlevel%==0 (
  gh repo create %USER%/%REPO% --public --source=. --remote=origin --push
  gh api -X PUT repos/%USER%/%REPO%/collaborators/jasminh
  gh api -X PUT repos/%USER%/%REPO%/collaborators/bkuehnis
) else (
  echo No gh CLI found. Create an empty repo named %REPO% at https://github.com/new then run:
  echo    git remote add origin https://github.com/%USER%/%REPO%.git
  echo    git push -u origin main
  echo Then add collaborators jasminh and bkuehnis under Settings ^> Collaborators.
)
