from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# メモリ上にデータを保存（再起動で消える）
todo_list = []

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "todos": todo_list})

@app.post("/add")
async def add_todo(task: str = Form(...)):
    todo_list.append(task)
    return RedirectResponse(url="/", status_code=303)