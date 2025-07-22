#!/usr/bin/env python3

import os, argparse, urllib, pathlib
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse


if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='simple static webserver')
    ap.add_argument('directory', type=str, default='static')
    ap.add_argument('port', type=int, default=1337)

    args = ap.parse_args()

    assert os.path.exists(args.directory)
    base_dir = pathlib.Path(args.directory)

    app = FastAPI()

    @app.get("/{file_path:path}")
    async def serve_files(file_path: str):
        full_path = base_dir / file_path

        if full_path.is_dir():
            file_path = '/'+file_path

            file_path_split = file_path.strip('/').split('/')
            if len(file_path_split) == 1 and file_path_split[0] == '':
                file_path_split = []

            path_links = []

            path_links.append(f'<a href="/">{base_dir}</a>')

            for i in range(len(file_path_split)):
                path_links.append(f'<a href="/{"/".join(file_path_split[:i+1])}">{file_path_split[i]}</a>')

            item_links = []

            if len(file_path_split):
                item_links.append(f'<li>📂 <a href="/{"/".join(file_path_split[:-1])}">..</a></li>')

            for item in sorted(os.listdir(full_path)):
                item_full_path = full_path / item
                item_url = '/'.join(file_path_split+[urllib.parse.quote(item)])
                if item_full_path.is_dir():
                    item_links.append(f'<li>📁 <a href="/{item_url}">{item}/</a></li>')
                else:
                    item_links.append(f'<li>📄 <a href="/{item_url}">{item}</a></li>')

            html_content = f"""
            <html>
                <head>
                    <title>Index of /{'/'.join(file_path_split)}</title>
                    <style>
                        body {{ font-family: sans-serif; padding: 20px; }}
                        a {{ text-decoration: none; color: #007BFF; }}
                        li {{ margin: 5px 0; }}
                    </style>
                </head>
                <body>
                    <h2>{'/'.join(path_links)}</h2>
                    <ul>
                        {''.join(item_links)}
                    </ul>
                </body>
            </html>
            """
            return HTMLResponse(content=html_content, status_code=200)

        elif full_path.is_file():
            return FileResponse(full_path)

        else:
            return HTMLResponse(content="404 Not Found", status_code=404)

    uvicorn.run(app, host='0.0.0.0', port=args.port)
