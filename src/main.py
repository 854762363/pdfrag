"""
FastAPI application entry point for PDFRAG.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.config import settings
from src.routes.upload import router as upload_router
from src.routes.query import router as query_router

app = FastAPI(
    title="PDFRAG",
    description="PDF Document Question Answering System with RAG",
    version="0.1.0",
)

app.include_router(upload_router)
app.include_router(query_router)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PDFRAG - 文档智能问答</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;color:#333}
.container{max-width:800px;margin:0 auto;padding:20px}
h1{text-align:center;margin:30px 0 10px;color:#1a1a2e}
.subtitle{text-align:center;color:#666;margin-bottom:30px}
.card{background:#fff;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.card h2{font-size:18px;margin-bottom:16px;color:#1a1a2e}
label{display:block;margin-bottom:6px;font-weight:500;font-size:14px;color:#555}
input,textarea,button{width:100%;padding:10px 14px;border:1px solid #ddd;border-radius:8px;font-size:14px;margin-bottom:12px}
textarea{resize:vertical;min-height:80px}
button{background:#4361ee;color:#fff;border:none;cursor:pointer;font-weight:600}
button:hover{background:#3a56d4}
button:disabled{background:#999;cursor:not-allowed}
.status{font-size:13px;margin-bottom:10px;min-height:20px}
.ok{color:#2d8a4e}.err{color:#d32f2f}
.answer-box{background:#f0f7ff;border:1px solid #cce5ff;border-radius:8px;padding:16px;margin-top:12px;white-space:pre-wrap;line-height:1.6}
.src-item{background:#fafafa;border:1px solid #eee;border-radius:6px;padding:10px;margin-top:8px;font-size:13px}
.src-meta{color:#888;font-size:12px}
.footer{font-size:12px;color:#999;text-align:center;margin-top:30px}
</style>
</head>
<body>
<div class="container">
<h1>📄 PDFRAG</h1>
<p class="subtitle">上传 PDF 文档，智能问答</p>
<div class="card">
<h2>📤 上传文档</h2>
<input type="file" id="fileInput" accept=".pdf">
<button id="uploadBtn" onclick="uploadPDF()">上传并处理</button>
<div id="uploadStatus" class="status"></div>
</div>
<div class="card">
<h2>💬 文档问答</h2>
<label>文档 ID</label><input type="text" id="docId" placeholder="自动填入">
<label>会话 ID</label><input type="text" id="convId" value="default">
<label>问题</label><textarea id="question" placeholder="输入你的问题..."></textarea>
<button id="chatBtn" onclick="askQuestion()">提问</button>
<div id="chatStatus" class="status"></div>
<div id="answerArea"></div>
</div>
<p class="footer">API: GET /health | POST /upload | POST /chat</p>
</div>
<script>
async function uploadPDF(){
const f=document.getElementById('fileInput').files[0];
const s=document.getElementById('uploadStatus');
if(!f){s.innerHTML='<span class=err>请选择 PDF 文件</span>';return}
s.textContent='上传中...';
const b=document.getElementById('uploadBtn');b.disabled=true;
const fd=new FormData();fd.append('file',f);
try{
const r=await fetch('/upload',{method:'POST',body:fd});
const d=await r.json();
if(d.status==='error')s.innerHTML='<span class=err>'+d.message+'</span>';
else{s.innerHTML='<span class=ok>✅ '+d.filename+' | '+d.pages+' 页 | '+d.chunks+' 块 | ID: '+d.doc_id+'</span>';document.getElementById('docId').value=d.doc_id}
}catch(e){s.innerHTML='<span class=err>上传失败</span>'}
b.disabled=false
}
async function askQuestion(){
const q=document.getElementById('question').value.trim();
const di=document.getElementById('docId').value.trim();
const ci=document.getElementById('convId').value.trim();
const s=document.getElementById('chatStatus');
const a=document.getElementById('answerArea');
if(!q){s.innerHTML='<span class=err>请输入问题</span>';return}
s.textContent='查询中...';
const b=document.getElementById('chatBtn');b.disabled=true;a.innerHTML='';
try{
const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,conversation_id:ci,doc_id:di||null})});
const d=await r.json();
s.textContent=d.confidence?'置信度: '+(d.confidence*100).toFixed(0)+'%':'';
let h='<div class=answer-box>'+d.answer.replace(/</g,'&lt;')+'</div>';
if(d.sources&&d.sources.length){h+='<h3 style=margin-top:16px>📚 来源</h3>';
d.sources.forEach(function(s){h+='<div class=src-item><strong>第'+s.page+'页</strong> '+(s.section||'').replace(/</g,'&lt;')+'<div class=src-meta>'+(s.content_preview||'').replace(/</g,'&lt;')+' | 相关度: '+(s.score*100).toFixed(0)+'%</div></div>'})}
a.innerHTML=h
}catch(e){s.innerHTML='<span class=err>查询失败</span>'}
b.disabled=false
}
</script>
</body>
</html>"""


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}
