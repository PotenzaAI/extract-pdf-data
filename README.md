# Conversor de PDF → Markdown com Supabase

Converte um PDF em **Markdown** com imagens, realiza **pós-processamento textual**, envia as **imagens para o Supabase Storage** e **reescreve o Markdown** para apontar para as **URLs públicas** dessas imagens. Durante o processo, **remove marcas d’água**, a **maioria das logos** e **imagens repetidas** (ex.: cabeçalhos/rodapés).
Pode operar lendo **registros de URL/ID** de uma **tabela no Supabase** e, ao final, enviar o **Markdown processado** para o próprio Supabase.


## Como funciona

### Leitura do PDF

Utiliza `pdf_to_md_docling_supabase.py` para converter o PDF em Markdown e extrair as imagens.
Cria um arquivo em markdown com o conteúdo da extração chamado `_docling.md`.

### Higienização de imagens

* Remove **marcas d’água** (via `--wm-clean`, utilizando `remove.py` por baixo dos panos).
* Remove **logos** conhecidas com base em um **template** (`--drop-image-template "logo.png"`) e **hash perceptual**.
* Deduplica **imagens repetidas** (ex.: cabeçalho/rodapé) com `--drop-repeated`.
* 
### Upload para Supabase Storage

* Envia as **imagens** para o **bucket** configurado.
* Reescreve o **Markdown** substituindo os caminhos locais pelas **URLs públicas** do Supabase em um novo arquivo markdown nomeado de acordo com o id `id.pdf` .

### Persistência

* Opcionalmente, lê entradas (**ID** e **URL**) de uma **tabela no Supabase** e, após processar, envia o **Markdown final** para o **bucket** configurado.

---

## Pontos de atenção

* Em **imagens compostas**, não foi possível remover algumas **logos automaticamente** (o corte poderia prejudicar o conteúdo). **Faça a limpeza manual** nesses casos.
* Há **imagens de eventos da CAB** e outras **referências à CAB** em imagens específicas; **remova manualmente** quando necessário.

---

## Requisitos

* **Python 3.10+**
* Dependências Python (instale com `pip install -r requirements.txt`)
* Uma conta e **projeto no Supabase** com:

  * **URL** do projeto
  * **Anon/Public Key** (ou chave adequada)
  * **Bucket** de storage criado (ex.: `articles-pdfs`)
* Arquivo de **logo de referência** para remoção (ex.: `logo.png`) na **raiz do projeto**
* Arquivo **`remove.py`** na **raiz** (utilizado para a limpeza de marca d’água)

> **Observação:** o script principal se chama `pdf_to_md_docling_supabase.py`.

---

## Configuração (env)

Crie um arquivo `.env` na raiz do projeto com:

```env
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_BUCKET=articles-pdfs

# Remoção de logo (template + pHash)
LOGO_HASHES=b897c360c66b9333
LOGO_PHASH_MAX_DIST=10

# Opcional, se houver uso de modelos externos em algum passo
OPENAI_API_KEY=
```

> **Segurança:** não faça commit do `.env` nem exponha `SUPABASE_KEY` publicamente.

---

## Execução

### Modo Supabase (lote pela tabela)

Processa registros lendo **ID** e **URL do PDF** de uma tabela no Supabase e envia o **Markdown final** e **imagens** ao Storage.

```powershell
# PowerShell (Windows)
python .\pdf_to_md_docling_supabase.py `
  --db-table transmission_manuals `          # Nome da tabela
  --db-id-col transmission_guide_id `                           # Nome da coluna com o ID
  --db-url-col pdf_url `                     # Nome da coluna com a URL do PDF
  --db-md-to storage `                       # Envia o Markdown ao bucket
  --storage-prefix "articles/images" `       # Caminho no bucket para IMAGENS
  --db-md-storage-prefix "articles/md" `     # Caminho no bucket para MARKDOWN
  --bucket articles-pdfs `                   # Nome do bucket
  --wm-clean `                               # Remove marcas d'água
  --drop-repeated `                          # Remove imagens repetidas (cab/rodapé)
  --drop-image-template "logo.png" `         # Template da logo (CAB) para remoção
  --emit-file --force-bom                    # Também gera o arquivo .md local e padrão utf-8
```

```bash
# Bash (Linux/macOS) - mesma chamada em múltiplas linhas:
python ./pdf_to_md_docling_supabase.py \
  --db-table transmission_manuals \
  --db-id-col transmission_guide_id \
  --db-url-col pdf_url \
  --db-md-to storage \
  --storage-prefix "articles/images" \
  --db-md-storage-prefix "articles/md" \
  --bucket articles-pdfs \
  --wm-clean \
  --drop-repeated \
  --drop-image-template "logo.png" \
  --emit-file --force-bom                   
```

### Modo URL única

Processa **apenas um** PDF indicado pela URL:

```powershell
# PowerShell (Windows)
python .\pdf_to_md_docling_supabase.py `
  --pdf-url "https://crm.cambioautomaticodobrasil.com.br/storage/tips/pdf/2635-09_g3_s_o_l_e_n_o_i_d_e_s.pdf" `
  --pdf-id "teste-final3_pdf" `
  --wm-clean `
  --drop-repeated `
  --drop-image-template "logo.png" `
  --emit-file --force-bom                   
```

```bash
# Bash (Linux/macOS)
python ./pdf_to_md_docling_supabase.py \
  --pdf-url "https://crm.cambioautomaticodobrasil.com.br/storage/tips/pdf/2635-09_g3_s_o_l_e_n_o_i_d_e_s.pdf" \
  --pdf-id "teste-final3_pdf" \
  --wm-clean \
  --drop-repeated \
  --drop-image-template "logo.png" \
  --emit-file --force-bom                    
```

---

## Parâmetros principais

* `--pdf-url <URL>`: URL de um PDF para processar (modo URL única).
* `--pdf-id <ID>`: identificador para nomear saídas (útil para versionar/organizar).
* `--db-table <nome>`: nome da tabela no Supabase (modo lote).
* `--db-id-col <coluna>`: coluna que contém o ID do registro (modo lote).
* `--db-url-col <coluna>`: coluna que contém a URL do PDF (modo lote).
* `--db-md-to storage|none`: se `storage`, envia o Markdown final para o Storage.
* `--storage-prefix <path>`: subpasta do bucket para **imagens**.
* `--db-md-storage-prefix <path>`: subpasta do bucket para **arquivos .md**.
* `--bucket <nome>`: bucket do Supabase (ex.: `articles-pdfs`).
* `--drop-image-template <arquivo>`: template de **logo** usado na remoção (ex.: `logo.png`).
* `--drop-repeated`: remove imagens repetidas típicas de cabeçalho/rodapé.
* `--wm-clean`: ativa o módulo de **limpeza de marca d’água** (usa `remove.py`).
* `--emit-file`: emite o **arquivo Markdown local** reescrito (além do envio ao Storage, se configurado).

---

## Dicas e Limitações

* **Imagens compostas:** quando a logo estiver muito mesclada com o conteúdo (texto/setas), a remoção automática pode degradar a imagem. **Edite manualmente** nesses casos.
* **Referências específicas da CAB:** algumas imagens de evento/branding podem **escapar** dos filtros e exigem **remoção manual**.
* **Hash & Distância (logo):** ajuste `LOGO_HASHES` e `LOGO_PHASH_MAX_DIST` no `.env` se trocar a arte de referência da logo.
* **Codificação:** garanta que o ambiente/gravação use **UTF-8** para evitar caracteres corrompidos no Markdown.

---

## Estrutura de pastas sugerida

```text
.
├── pdf_to_md_docling_supabase.py
├── remove.py
├── logo.png
├── .env
├── requirements.txt
└── README.md
```

---

## FAQ rápido

**1) Preciso do `remove.py` mesmo usando `--wm-clean`?**
Sim. Ele é utilizado na etapa de remoção de marca d’água; mantenha-o na raiz do projeto.

**2) Onde ficam as imagens e o Markdown no Supabase?**
No **bucket** definido em `SUPABASE_BUCKET`, nas subpastas informadas por `--storage-prefix` (imagens) e `--db-md-storage-prefix` (Markdown), com **URLs públicas** reescritas no `.md`.

**3) Posso rodar sem Supabase (somente local)?**
Sim. Use o modo de **URL única** e **omite** os parâmetros de envio ao storage. Com `--emit-file`, o Markdown final será salvo localmente.
