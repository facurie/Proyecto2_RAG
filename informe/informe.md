# Retrieval & Memory Lab · Informe

**Proyecto 2 · IA Generativa Avanzada · UTDT.** Asistente de RAG con memoria sobre legislación española de vivienda y alquiler (BOE, legislación consolidada).

*Basado en datos de la Agencia Estatal Boletín Oficial del Estado. Los textos consolidados tienen carácter meramente informativo: solo la publicación en el diario oficial tiene validez jurídica.*

**Cómo está organizado este informe.** Cada sección corresponde a una etapa y remite a su notebook (`NN_*.ipynb`, con salidas en el zip), donde está cada tabla junto con su lectura. Trabajamos sobre los scripts `.py` de cada etapa (`01_documents.py` a `07_second_read.py`) y generamos los notebooks a partir de ellos con `tools/py_to_notebook.py`. Las decisiones, con su razón y la alternativa que descartamos, están en `decision_log.md`. Salvo que se diga otra cosa, los números son sobre el split `test`. Como el conjunto es chico, cada comparación hay que leerla contra su cota de ruido: con n = 30, una fila tiene unos ±12 puntos de margen.

El código está en el repo público https://github.com/facurie/Proyecto2_RAG. La primera celda de cada notebook lo clona e instala las dependencias en Colab.

---

## 1 · Decisiones de diseño

Antes de entrar en las etapas conviene explicar las decisiones que más pesan en los resultados, porque varias atraviesan todo el proyecto. El razonamiento completo de cada una está en `decision_log.md`.

La primera tiene que ver con la naturaleza del corpus. El BOE consolidado guarda cada versión de cada artículo junto con su período de vigencia. Eso es lo que hace interesante a este corpus, porque permite armar las mejores preguntas temporales, pero también es su riesgo principal: la fuente cambia mientras uno trabaja sobre ella. Por eso fijamos una foto del corpus identificada por un hash (`corpus_sha`) y congelamos el eval set contra esa foto. De ahí sale también que el gold esté definido a nivel de sección (una versión de un artículo) y no de documento. Una pregunta sin fecha tiene como gold la versión vigente el día de la foto, y una pregunta con fecha, la versión vigente ese día. Si el gold fuera la norma entera, no podríamos distinguir un sistema que recupera la versión correcta de uno que recupera la equivocada.

La segunda decisión fue usar una sola vara de medir en todas las etapas. La entrega de pasajes por presupuesto de tokens (`retrieval.deliver`), la definición de cuándo "le llegó la respuesta" al generador (`evaluation.delivered_metrics`, que exige que llegue un pasaje del gold que contenga la clave) y el puntaje sin juez (`grounded.score`) se escribieron una sola vez y los usan las Etapas 3 a 6. Parece un detalle de implementación, pero es lo que permite comparar números entre etapas. Si cada etapa definiera "acierto" a su manera, una diferencia entre la Etapa 3 y la 4 podría ser solo una diferencia de definición.

También decidimos tratar los errores como una categoría propia. Una llamada que falla, ya sea por overflow, por un problema de parseo, por truncamiento o porque la gramática cortó la respuesta en su tope, nunca cuenta como abstención ni como respuesta. Si las mezcláramos, la tasa de abstención quedaría inflada con fallas que no son decisiones del modelo. En la misma línea, la abstención es el booleano del schema (`can_answer`), y el veredicto de las preguntas de sí/no se lee de la primera palabra de `answer`, solo en esas preguntas y solo si el modelo dijo que podía responder.

Otra decisión fue no confiar en nuestra propia cuenta de tokens sin verificarla. En cada llamada fijamos `num_ctx = 8.192` y `num_predict = 768`, y comparamos la cuenta que hacemos con la plantilla de chat contra el `prompt_eval_count` que devuelve el servidor. La diferencia resultó constante, de 4 tokens. Puede parecer una precaución exagerada, pero después fue lo único que nos permitió detectar un prompt truncado en silencio (sección 5).

Por último, dos decisiones de método. El ruido lo acotamos por la unidad que es realmente independiente: la pregunta en las Etapas 3 y 4, el guion en la 5 y el escenario en la 6, con test de signos y bootstrap por grupo. Contar turnos o sondas como si fueran independientes daría intervalos más angostos de lo que corresponde, porque los turnos de un mismo guion dependen unos de otros. Además, no paralelizamos las corridas. Tenemos una sola GPU, cada experimento usa un único `num_ctx`, y la salida depende del estado de la caché de prompts de Ollama, así que cada brazo arranca con el modelo descargado y en un orden fijo.

## 2 · El corpus (Etapas 0 y 1)

Elegimos 177 normas de vivienda y alquiler, estatales y autonómicas, con una regla de densidad temática sobre 10 materias del BOE (la regla completa está en la ficha, al final de esta sección). Después de la extracción quedan 8.686 secciones, es decir, versiones de artículos, de las cuales 5.766 están vigentes el día de la foto. En tokens del generador son 4,78 M con todas las versiones y 2,89 M de texto vigente, así que el corpus no entra en la ventana: de las 135 normas vigentes, solo 47 entran solas en un prompt de 7.424 tokens.

La foto que usamos es la del 2026-10-24, no la original. La primera, del 2026-10-16, se perdió entre máquinas y no se podía volver a bajar igual, porque en esos ocho días se habían actualizado 6 de las 177 normas, entre ellas la Ley 12/2023 y el Plan Estatal de Vivienda. Rehicimos la Etapa 0 con la misma regla, que devolvió las mismas 177 normas, y volvimos a validar el eval set. Falló un solo gold: el art. 10 de la LPH tenía una versión nueva, vigente desde el 2026-10-23, con la misma respuesta. Lo corregimos y congelamos el eval set de nuevo, antes de mirar cualquier resultado (`decision_log.md`, sección 5). El episodio fue molesto, pero mostró en la práctica algo que la Etapa 0 decía en teoría: este corpus cambia lo suficientemente rápido como para romper un gold en una semana.

Gran parte del trabajo de estas dos etapas fue encontrar datos sucios que no tiran ningún error. Los más importantes fueron cuatro: la conversión a JSON de la API pierde todas las materias menos la última; el texto citado por las leyes modificadoras parecía una nota editorial; la versión vigente de un artículo no siempre es la última; y 434 secciones tienen un gemelo casi literal en otra norma. Los detallamos en las respuestas que siguen y en la ficha.

### Preguntas de la Etapa 0

**¿Qué propiedades no cumple el corpus, cuánto cuesta la octava y en qué etapa se paga cada una?**

De las siete propiedades, el corpus falla en tres. La primera es que el texto no está limpio para un loader ingenuo: hay versiones superpuestas, notas de modificación metidas dentro del texto, tablas, y el JSON que pierde materias. Ese costo lo paga la Etapa 1.

La segunda es que los metadatos a veces mienten. Una ley balear figura como vigente cuando su propio texto dice que está derogada, las resoluciones anuales ya reemplazadas siguen figurando vigentes, y `fecha_publicacion` es la fecha de la copia en el BOE y no la del boletín original. Esto también lo paga la Etapa 1, con un mapeo de estado corregido, pero lo heredaría cualquier filtro por fecha o por región en las Etapas 3 y 6, así que no es un problema que se resuelva una vez y se olvide.

La tercera es que el corpus cambia mucho: el 18% de los preceptos tiene más de una versión, y la LAU cambió tres veces solo en octubre de 2026. Lo pagamos con la foto fijada en la Etapa 1, pero al mismo tiempo lo aprovechamos, porque es lo que permite hacer preguntas "a una fecha" y actualizaciones de memoria en las Etapas 2 y 6.

Las otras cuatro propiedades se cumplen. La respuesta está en el corpus, con huecos conocidos (la LEC y la ley catalana, que usamos para las preguntas fuera del corpus). Hay estructura para cortar, con bloques tipados y una mediana de 230 tokens por precepto. Los ids son estables, y se puede armar un eval set.

La octava, la similitud, es alta, y es probablemente la propiedad que más define este proyecto. El 7% de los artículos tiene un gemelo casi literal en otra norma, y las versiones consecutivas de un mismo artículo tienen una similitud de Jaccard mediana de 0,82. O sea que dos textos casi iguales pueden dar respuestas distintas. Esto se paga en tres lugares: en la Etapa 1, que trata las versiones como versiones y no como copias; en la Etapa 2, donde el gold tiene que llevar fecha o ser un conjunto; y en la Etapa 3, donde el prefijo contextual incluye la vigencia para que el recuperador pueda distinguirlas.

**¿Entraría el subconjunto en la ventana? Y si entrara, ¿para qué construir recuperación?**

No entra. Son 2,89 M tokens vigentes (4,78 M si se cuentan todas las versiones) contra 7.424 tokens de prompt. Hay 47 normas vigentes, de 135, que entran solas, y para esas la estrategia de "si entra, pegalo" sería posible. Aun así, creemos que la recuperación seguiría haciendo falta, porque en este corpus no alcanza con que llegue la norma correcta: tiene que llegar la versión correcta. Una pregunta "a una fecha" necesita la versión de ese día, que muchas veces no es la que está en el texto vigente. Y la Etapa 4 muestra que, cuando el modelo tiene a la vista dos versiones del mismo artículo, aplica la equivocada (md02, md05), así que pegar la norma con todas sus versiones tampoco lo resuelve. Hay que aclarar que la fila "norma entera" no la corrimos, así que esto es un argumento y no un resultado medido.

**¿Cuánto del corpus pudo haber leído el generador, y cómo se distingue recuperación de recuerdo?**

El 11% de los preceptos nació después de mayo de 2025, que es el corte de Qwen3, y 18 normas se publicaron después de esa fecha. Esas, el modelo no las pudo leer. Lo anterior puede haberlo leído, pero en la práctica no lo recuerda: a libro cerrado acierta 1 de 5 en el chequeo de la Etapa 0 y 3 de 26 en la fila de libro cerrado de la Etapa 4, y muchas veces se equivoca con alucinaciones seguras. Esa fila de libro cerrado es justamente la que separa las dos cosas. Lo que se acierta sin pasajes es recuerdo, y la diferencia con la fila de recuperación (+43 puntos) es lo que aporta recuperar. Que el modelo sepa tan poco de este corpus es una buena noticia para el experimento, porque casi todo lo que acierta con recuperación se lo podemos atribuir a los pasajes.

**¿Qué hace que dos documentos se confundan, y conviene conservarlo?**

Encontramos cuatro fuentes de confusión. La más importante son las versiones de un mismo artículo, que tienen casi el mismo texto y distinta respuesta. Después están los decretos-ley derogados y vueltos a decretar (RDL 21/2018 y 7/2019, o 26 y 29/2026), el decreto-ley y la ley que lo convalidó, y las 13 resoluciones anuales escritas con la misma plantilla.

Decidimos conservar todo esto, porque es exactamente lo que el corpus pone a prueba: las preguntas a una fecha, el filtro de fecha y las actualizaciones de memoria dependen de que esas confusiones existan. Limpiarlas habría hecho el problema más fácil, pero también menos parecido al uso real de una base legal. Por eso no se borra ni se fusiona nada, y el gold apunta a la versión exacta o al conjunto de gemelos vigentes.

### Preguntas de la Etapa 1

**¿Qué cambió la limpieza, en números, y qué regla cambió más?**

La limpieza llevó el corpus de 5,14 M tokens, que es lo que daría un loader ingenuo, a 4,78 M en `documents.jsonl`, y a 2,89 M si se cuenta solo el texto vigente. La regla que más cambió fue la que saca las notas de modificación a `Section.notes`, que movió 848.500 caracteres en 116 normas. Muy por detrás le siguen la firma (33.159 caracteres en 172 normas) y la fórmula de promulgación (17.580 en 77), mientras que la linealización de tablas agrega 89.413 caracteres. A nivel de caracteres, en cambio, el XML es limpio: la normalización NFC toca solo 2 normas y los caracteres invisibles, 5.

La conclusión que sacamos es que el problema de este corpus no es de codificación sino de estructura. Lo difícil no fue limpiar caracteres raros, sino decidir qué parte del texto es la norma y qué parte es un comentario sobre ella. La tabla por regla está en la ficha.

**Un defecto que un loader ingenuo habría dejado pasar sin error, y cómo se encontró.**

El más claro es el de las materias. La conversión a JSON del análisis de la API conserva solo la última materia de cada norma, de modo que 436 de 438 normas pierden materias y la LAU, que tiene 17, queda con una sola. No aparece ningún error ni advertencia. Lo encontramos al comparar las dos respuestas que da la API para la misma norma, porque en XML la LAU sí tiene sus 17 materias, y desde entonces leemos el análisis en XML. Es un buen ejemplo de por qué conviene cruzar fuentes: el JSON era perfectamente válido, solo que estaba incompleto.

Encontramos otro caso parecido con el texto que cita una ley modificadora. Viene en un `<blockquote>`, igual que las notas, y al descartarlo junto con ellas el "Artículo único. Se modifica la Ley X, que queda redactada así:" quedaba vacío, justo en el lugar donde está el contenido nuevo.

**¿Cómo se definieron documento y sección, y qué se rompe si una etapa posterior no está de acuerdo?**

Un documento es una norma, y una sección es un bloque del BOE en una de sus versiones, con id `doc_id#bloque@AAAAMMDD` y una vigencia `[valid_from, valid_to)`. Lo importante es que la fecha forma parte del id. Si una etapa posterior no respeta esta definición, se rompen dos cosas. Un gold sin fecha (`doc_id#bloque`) pasa a ser ambiguo entre versiones, y un índice que ignore `valid_to` le entrega al generador una versión que ya no rige o que todavía no rige. El segundo error es especialmente peligroso porque no produce ninguna falla visible: el modelo responde igual, solo que con la ley equivocada.

**¿Sobrevivirían los `doc_id` a una descarga nueva?**

Sí, y de hecho lo pudimos comprobar. El `doc_id` es el identificador oficial del BOE, y el `section_id` lleva el día desde el que rige la versión, de modo que una versión nueva agrega un id en lugar de cambiarle el contenido a uno que ya existía. Si hubiéramos usado solo `doc_id#bloque`, los arts. 38-40 de la LAU, derogados desde 2001, habrían cambiado de contenido en octubre de 2026, cuando se llenaron con un régimen nuevo.

La prueba real fue la foto nueva. Con 6 normas actualizadas, el eval set validó entero salvo un gold, cuyo id seguía existiendo pero había dejado de ser la versión vigente. Los ids sobrevivieron; lo que cambió fue cuál de ellos rige hoy, que es exactamente lo que el diseño prevé.

**¿Cómo se trataron los casi-duplicados, y a qué compromete eso en el eval set?**

No borramos ni fusionamos nada. De los 90 pares de normas parecidas, ninguno resultó ser una copia: 87 son versiones (una reemplaza a la otra, y la reemplazada lleva `superseded_by`) y 3 son normas relacionadas que rigen las dos. A nivel de sección, 434 tienen un gemelo casi literal en otra norma, y en 65 de esos pares rigen los dos.

Esto compromete al eval set a dos cosas. El gold de una pregunta sin fecha es la versión vigente más sus gemelos vigentes, tomados como conjunto, y el de una pregunta a una fecha es la versión de ese día. Lo primero evita que el sistema sea penalizado por recuperar un gemelo que dice exactamente lo mismo que el gold.

### La ficha del corpus (corpus card)

Tiene el mismo contenido que `corpus_card.md`. Los números son los de la foto del 2026-10-24 y salen de `01_documents.ipynb`.

#### Fuente y versión fijada

La fuente es la API de datos abiertos del BOE, colección de legislación consolidada (`https://www.boe.es/datosabiertos/api/legislacion-consolidada`), documentada en `https://www.boe.es/datosabiertos/documentos/APIconsolidada.pdf` (versión del 2 de septiembre de 2025). Para cada norma trae los metadatos (rango, ámbito, departamento, fechas, vigencia, estado de consolidación, ELI), el análisis (materias, notas, referencias a normas anteriores y posteriores) y el texto consolidado con **todas las versiones fechadas** de cada bloque, cada una con su `fecha_vigencia` y la norma que la introdujo.

La versión fijada es la descarga del **2026-10-24**. La del 2026-10-16 se perdió y no se podía volver a bajar, porque 6 normas habían cambiado. Las respuestas crudas están en `data/raw/boe_cache/` (fuera de git), con la fecha de descarga y el sha256 de cada archivo en `index.jsonl`, y la lista de normas con la `fecha_actualizacion` de cada una está en `data/subset_v1.json` (en git). La fuente cambia seguido: la LAU tuvo versiones nuevas el 1, el 2 y el 8 de octubre de 2026. Para ver qué se actualizó desde la foto alcanza con consultar `?from=20261024`.

En la Etapa 1 verificamos la foto antes de usarla. Son 1.763 archivos y ninguno falta ni difiere de su sha256 (la comparación se hace sobre el texto, porque la foto del 2026-10-16 se había guardado con CRLF en Windows). Las 177 normas siguen en la `fecha_actualizacion` fijada, y el build lee solo la caché, sin conexión. Todo queda resumido en el hash del corpus, `corpus_sha = b01a2c0acc3508873861365d25b37d28ea0a8dacf7f868834ddb5207e28b526d`, guardado en `artifacts/full/documents.manifest.json`. Un build en otra máquina (Colab) tiene que dar el mismo.

El idioma es español jurídico peninsular. Las normas autonómicas están en su versión castellana publicada en el BOE.

#### Licencia y atribución

La reutilización se rige por la Ley 37/2007 de reutilización de la información del sector público, que permite el uso comercial y no comercial. El aviso legal del BOE pone tres condiciones: no desnaturalizar el sentido de la información, no sugerir carácter oficial ni patrocinio del BOE, y conservar los metadatos y las fechas de actualización.

Las obras derivadas llevan la atribución "Basado en datos de la Agencia Estatal Boletín Oficial del Estado". Además, en cada texto reutilizado hay que indicar que los textos consolidados tienen **carácter meramente informativo** y que solo la publicación en el diario oficial tiene validez jurídica; por eso ese aviso va en el encabezado del informe. Los datos crudos no se commitean (assignment §6.1).

#### Regla del subconjunto

1. **Universo:** normas consolidadas de 10 materias de vivienda del vocabulario controlado del BOE: Arrendamientos (303), Arrendamientos urbanos (305), Desahucios (2476), Propiedad Horizontal (5774), Viviendas (7162), Viviendas de Protección Oficial (7163), Viviendas de Renta Limitada (7164), Viviendas sociales (7166), Impuesto sobre Viviendas Desocupadas (4118), Cooperativas de viviendas (1720). Son **438 normas y 16,1 M tokens**.
2. **Densidad temática ≥ 0,45:** al menos el 45% de los tokens del articulado vigente está en artículos que mencionan vivienda, alquiler, desahucio o propiedad horizontal (regex en `data/subset_v1.json`). Esto saca las normas ómnibus (presupuestos, medidas fiscales, LEC, Impuesto sobre Sociedades).
3. **`estado_consolidacion = Finalizado`:** saca 15 normas autonómicas cuyo texto el BOE marca como desactualizado.

**Resultado: 177 normas y 3,09 M tokens** del generador (tokenizer de `Qwen/Qwen3-8B`, contando la última versión de cada bloque, como en la Etapa 0). Con la extracción de la Etapa 1 son **4,78 M tokens con todas las versiones** y **2,89 M de texto vigente el 2026-10-24**. La diferencia con los 3,09 M tiene tres causas: la última versión no siempre es la que rige, las normas derogadas no tienen nada vigente, y el texto citado por las leyes modificadoras ahora sí se cuenta. Con la extracción corregida ninguna norma cae debajo del umbral de densidad.

| | Vigentes | Agotadas | Total |
|---|---|---|---|
| Estatales | 81 | 11 | 92 |
| Autonómicas | 67 | 18 | 85 |
| **Total** | 148 | 29 | 177 |

Por rango hay 73 leyes, 22 decretos-ley, 22 reales decretos, 16 reales decretos-ley, 16 órdenes, 14 resoluciones y otros. El tamaño varía mucho. En la cuenta de la Etapa 0, los tokens por norma tienen un p10 de 2.196, una mediana de 10.469, un p90 de 45.113 y un máximo de 121.280, y 76 de las 177 normas entran solas en un prompt de 7.424 tokens. Si se mira solo el texto vigente de las normas vigentes (Etapa 1), los valores son p10 1.467, mediana 12.641, p90 51.862 y máximo 123.867, y **47 de las 135** entran solas en 7.424 tokens; para esas preguntas, "si entra, pegalo" sería posible. En total hay 5.468 preceptos, con una mediana de 230 tokens.

Como control de recall tomamos las 13 normas estatales con "alquiler" en el título: 12 están en la unión de materias, y la que falta (RD 933/2021, registro de hospedajes) está bien afuera.

#### Documento y sección

Un **documento** es una norma. Su `doc_id` es el identificador oficial del BOE (`BOE-A-1994-26003`, `BOCL-h-2020-90108`, ...), que es estable entre descargas.

Una **sección** es un bloque del BOE en una de sus versiones. El bloque puede ser un artículo, el preámbulo, un título, un anexo, la nota inicial o la parte dispositiva; las firmas se descartan. El id tiene la forma `section_id = doc_id#bloque@AAAAMMDD` (por ejemplo, `BOE-A-1994-26003#a9@20130606`), con el id de bloque del BOE y el día desde el que rige esa versión. El id de bloque **no** se deriva del título (el art. 9 bis es `a9-2`), y si dos versiones del mismo bloque caen el mismo día, la segunda lleva `.2`.

Cada sección rige en el intervalo `[valid_from, valid_to)`, y `Section.in_force(fecha)` dice si rige en una fecha dada. La versión vigente no siempre es la última: el art. 10 de la LAU tiene una versión que rige recién desde el 15/11/2026. `block_id` agrupa las versiones de un mismo bloque, y `path` es la miga de pan, es decir, los títulos y capítulos que contenían a esa versión el día en que empezó a regir (las disposiciones adicionales, transitorias y finales no cuelgan de ninguno).

`Document.text` es la concatenación de las secciones separadas por una línea en blanco, con un párrafo por línea y ya normalizado. Las notas de modificación y las editoriales quedan en `Section.notes`, fuera del texto. En total son 177 documentos y 8.686 secciones, de las que 5.766 rigen el 2026-10-24; hay 6.735 bloques, y 1.032 tienen más de una versión.

Si una etapa posterior no está de acuerdo con esta definición, se rompen dos cosas: un gold escrito como `doc_id#bloque`, sin fecha, es ambiguo entre versiones, y un índice que no respete `valid_to` le entrega al generador una versión que ya no rige o que todavía no rige.

#### Mapeo de estado

`DocMetadata.status` solo acepta `current`, `superseded`, `deprecated`, `draft` o `unknown`. Los campos crudos (`vigencia_agotada`, `estatus_derogacion`, `estatus_anulacion`, `estado_consolidacion`) quedan en `source_status`, y la razón de cada corrección en `status_reason`.

| Fuente | `status` | Normas |
|---|---|---|
| `vigencia_agotada = N` | `current`, salvo las correcciones de abajo | 148 → 135 |
| `vigencia_agotada = S` (derogada, agotada por plazo o sustituida tácitamente) | `superseded` | 29 |
| **Correcciones verificadas contra el texto** (`corpus.CORRECTIONS`, `corpus.PERIODIC_SERIES`) | | |
| Ley balear 11/2005 (`BOE-A-2005-21319`): figura vigente, pero su nota inicial dice "Norma derogada, con efectos de 1 de enero de 2018" | `superseded`, cerrada el 2018-01-01 | 1 |
| Resoluciones anuales de tipos de interés de préstamos convenidos (13, de 2014 a 2026): todas figuran vigentes | `current` solo la de 2026; las otras 12, `superseded`, con `superseded_by` a la siguiente | 12 |
| RDL 27/2012 (`BOE-A-2012-14115`): agotada sin fecha de derogación | `superseded`, cerrada el 2013-05-15 (entrada en vigor de la Ley 1/2013, que lo sustituyó) | — |

**Resultado: 135 `current` y 42 `superseded`.** De las 42, 37 llevan `superseded_by`; las otras 5 no nombran a la norma que las reemplazó. Cuando una norma deja de regir, sus versiones se cierran en esa fecha, así que después de su fin no rige nada, salvo 3 versiones transitorias que el BOE fecha después de la derogación (`after_norm_end`).

También medimos cuánto discrepan los metadatos con el cuerpo de las normas. El estado contradice a la nota inicial en 1 de 177 normas (la ley balear). La `fecha_vigencia` coincide con la cláusula "entrará en vigor el día siguiente al de su publicación" en las 57 normas que se pueden chequear (57/57), siempre que se tome la publicación oficial. `fecha_publicacion`, en cambio, falla en 17 de las 22 autonómicas chequeables, las de id `BOE-A`, porque es la fecha de la copia en el BOE. Por eso los filtros por fecha usan `valid_from`/`valid_to` y nunca `fecha_publicacion`.

#### Defectos conocidos

Esta es la lista de lo que un loader ingenuo dejaría pasar sin ningún error.

1. **El JSON del análisis pierde materias.** La conversión de la API conserva solo la última `<materia>`, así que 436 de 438 normas pierden materias (la LAU queda con 1 de 17). Por eso el análisis se lee en XML.
2. **Versiones superpuestas.** El texto trae todas las versiones de cada bloque. Concatenarlas lleva el subconjunto de 3,09 M a 4,46 M tokens y, lo que es peor, mezcla reglas contradictorias: el art. 9 de la LAU dice "cinco años", "tres años" y otra vez "cinco años".
3. **Notas de modificación dentro del texto, y un `<blockquote>` que no es nota.** Las notas ("Se modifica el apartado 3 por...", "Téngase en cuenta...") son 2.685 `<blockquote>`, unos 0,36 M tokens, y van a `Section.notes`. Pero los 748 `<blockquote class="sangrado">` (unos 0,31 M tokens en 38 normas) son el texto que cita una ley modificadora. La auditoría de la Etapa 0 los descartaba junto con las notas, y el "Artículo único. Se modifica la Ley X así:" quedaba vacío. En la Etapa 1 se tratan como texto.
4. **Metadatos que mienten o que significan otra cosa.** `vigencia_agotada` coincide con `estatus_derogacion` en 176 de 177 normas; la que no coincide está agotada sin derogación (una sustitución tácita). Contra el texto, una norma figura vigente y su texto dice que está derogada, y las normas periódicas reemplazadas siguen figurando vigentes. En la unión de materias (no en el subconjunto) hay además un caso curioso: la nota del RDL 2/2026 dice "Norma *deroga* por..." (sic), así que una búsqueda de "derogada" no la encuentra.
5. **`fecha_publicacion` no es la fecha oficial en las autonómicas con id `BOE-A`.** Es la de la copia en el BOE, posterior a la del boletín regional, y en 64 normas autonómicas es posterior incluso a la entrada en vigor. La fecha regional está en una nota del análisis ("Publicada en el DOG núm. 88, de 10 de mayo de 1988"). Contra la cláusula "entrará en vigor el día siguiente al de su publicación", `fecha_publicacion` falla en 17 de 17 de esas normas, y la fecha regional acierta en las 17. Lo que corresponde usar es `fecha_vigencia` (`valid_from`).
6. **La misma región con dos nombres** de departamento: "Comunitat Valenciana" (5 veces) y "Comunidad Valenciana" (1).
7. **Un mismo id de bloque puede cambiar por completo de contenido.** Los arts. 38-40 de la LAU estaban derogados desde 2001, y en octubre de 2026 se llenaron con un régimen sancionador para plataformas de alquiler de corta duración.
8. **Formato.** El 23% de las normas tiene tablas: 166 fuera de las firmas, que linealizamos, aunque las fórmulas maquetadas como tabla (Kt = 0,35 · Ht/Ho + ...) quedan ilegibles igual. Hay 162 imágenes (URLs a PNG, no base64) en 29 normas, que reemplazamos por `[imagen omitida]`, así que lo que dicen no está en el texto. El 3% de los preceptos supera los 1.500 tokens, y los "Artículo único" de las leyes modificadoras llegan a 25 K. En cuanto a caracteres, "m2" y "m²" conviven entre versiones. NFKC los unifica, pero también cambia "1.º" por "1.o" (2.911 ordinales) y rompe la prima de los subapartados "a´)", por lo que unificamos solo superíndices y ligaduras. Por último, el texto de una celda con `<br/>` pegaba palabras ("personasque"), lo que ya está corregido.
9. **La versión vigente no es la última.** Pasa en 3 bloques; por ejemplo, el art. 10 de la LAU tiene una versión del RDL 28/2026 que rige recién desde el 15/11/2026. Las fechas tienen además otras irregularidades: 18 versiones vienen sin `fecha_vigencia` y se fechan con la norma que las introdujo o, si no, con su publicación; 11 bloques (54 versiones) traen las versiones fuera de orden; 6 versiones se reemplazan el mismo día; 23 caducaron antes de empezar a regir; 38 empiezan el mismo día en que termina su norma, y 3 están fechadas después de la derogación de su norma (régimen transitorio).
10. **La caché cruda no coincide en bytes con su sha256.** Se guardó en Windows con CRLF, aunque el contenido sí coincide, así que la verificación compara el texto decodificado.

**Huecos de cobertura.** Quedaron afuera la ley catalana 18/2007 del derecho a la vivienda y otras 14 autonómicas (desactualizadas en el BOE, y por eso excluidas), y el procedimiento de desahucio de la LEC (excluida por densidad, 0,13).

#### Limpieza (Etapa 1)

La tabla muestra, para cada regla, en cuántas normas cambió algo y cuántos caracteres sacó (un número negativo significa que agregó). El tratamiento de cada elemento y su porqué están en `decision_log.md` (Etapa 1).

| Regla | Qué hace | Normas | Caracteres |
|---|---|---|---|
| `notes_out` | notas de modificación y editoriales (`<blockquote>`) → `Section.notes` | 116 | 848.500 |
| `signature` | descarta el bloque de firma | 172 | 33.159 |
| `promulgation` | descarta la fórmula de promulgación del preámbulo | 77 | 17.580 |
| `editorial_inline` | notas "(\*) Téngase en cuenta" en línea → `notes` | 5 | 1.491 |
| `whitespace` | espacios repetidos dentro de una línea | 5 | 41 |
| `invisibles` | guion blando, ancho cero, BOM, control | 5 | 33 |
| `nfc` | acentos descompuestos → compuestos | 2 | 17 |
| `odd_spaces` | NBSP y espacios tipográficos → espacio común | 92 | 0 |
| `sup_ordinals` | `5.<sup>o</sup>` → `5.º` | 3 | 0 |
| `compat` | superíndices y ligaduras (² → 2) | 13 | −1 |
| `images` | imagen → `[imagen omitida]` | 29 | −2.592 |
| `tables` | tablas linealizadas ("Encabezado: valor; ...") | 41 | −89.413 |

En tokens del generador:

| Versión del texto | Tokens |
|---|---|
| Lo que daría un loader ingenuo (todas las versiones, con notas y firmas) | 5,14 M |
| `documents.jsonl` (todas las versiones) | 4,78 M |
| Texto vigente el 2026-10-24 | 2,89 M |

Como ya dijimos, lo que más cambia es estructural (notas, versiones, texto citado) y no de codificación. A nivel de caracteres, el XML del BOE es limpio.

#### Casi-duplicados: política

Buscamos casi-duplicados en tres niveles. Entre normas comparamos todos los pares (con Jaccard ≥ 0,3 o contención ≥ 0,8) y encontramos 90, de los cuales ninguno es una copia (Jaccard ≥ 0,9). En 87 de ellos una de las dos normas está reemplazada, así que son versiones: un RDL y su ley de convalidación, un RDL derogado y el que lo re-decretó, o una resolución anual y la siguiente. Los otros 3 son normas relacionadas que comparten texto y rigen las dos. Todos quedan registrados en `metadata.near_duplicates`, la reemplazada lleva `superseded_by`, y no se borra ni se fusiona ninguna.

Entre secciones de normas distintas (Jaccard ≥ 0,7), 434 secciones tienen un gemelo, guardado en `Section.twins`, y en 65 de esos pares rigen los dos el 2026-10-24. Las versiones de un mismo bloque, en cambio, las tratamos como secciones distintas, porque dan respuestas distintas.

Para la Etapa 2, esto se traduce en que el gold de una pregunta sin fecha es la versión vigente más sus gemelos vigentes, como conjunto, y el de una pregunta a una fecha es la versión vigente en esa fecha.

#### Qué pone a prueba bien este corpus

Creemos que este corpus pone a prueba bien seis cosas. La primera son las **versiones con respuestas distintas**. El 18% de los preceptos tiene más de una versión (el 8%, tres o más), y las versiones consecutivas son casi idénticas en texto (Jaccard mediana 0,82) pero distintas en la respuesta. Eso permite hacer preguntas "a una fecha" (un contrato firmado en 2016 tenía 3 años de plazo mínimo, no 5) y actualizaciones de memoria.

La segunda son los **vecinos confundibles**. El 7% de los artículos tiene un gemelo casi literal en otra norma; en la Etapa 1, contando versiones, son 434 secciones, y en 65 pares rigen los dos a la vez. A eso se suman los pares de RDL derogado y RDL re-decretado (21/2018 y 7/2019; 26 y 29/2026; 27 y 28/2026), los decretos-ley y las leyes que los tramitaron, y las 13 resoluciones anuales con la misma plantilla.

También hay **identificadores literales**, como números de ley y de artículo, que sirven para la categoría `identifier` y para el canal léxico. El corpus permite además separar **recuperación de recuerdo**: el 11% de los preceptos nació después de mayo de 2025 (el corte de Qwen3) y 18 normas se publicaron después, entre ellas el Plan Estatal de Vivienda 2026-2030, la Ley 5/2025 de Vivienda de Andalucía y los RDL de octubre de 2026, que el generador no pudo leer. La dimensión de **región** está cubierta por 85 normas autonómicas de 16 comunidades, útiles para el filtro de metadatos y para la memoria del usuario ("vivo en Baleares"). Por último, lo excluido (LEC, ley catalana) y lo derogado dan preguntas `out_of_corpus` plausibles, lo que hace que la **abstención** sea difícil: la pregunta parece del tema aunque la respuesta no esté.

#### Chequeo a libro cerrado

En `notebooks/00_closed_book.ipynb` hicimos 5 preguntas sin recuperación, una por categoría más una posterior al corte. En la corrida local del 2026-10-16, con el modelo al 63% en GPU, acertó **0/5**, con 2 abstenciones y 3 alucinaciones seguras. Repetida el 2026-10-24 con el modelo 100% en GPU (Ollama 0.34.4), acertó **1/5**. El resultado se mantiene con dos variantes de formato: el schema sin la instrucción de sí/no, y texto libre.

`qwen3:8b` no conoce ni siquiera los datos más difundidos de este corpus, como la prórroga de la LAU, la fianza del art. 36 o el umbral de gran tenedor. Por eso un eval set sobre este corpus mide recuperación y no recuerdo.

## 3 · El conjunto de evaluación

El conjunto tiene tres partes. Hay 40 preguntas de un turno (30 de test: 13 `topical`, 6 `identifier`, 7 `metadata` y 4 fuera del corpus), 10 guiones de conversación (8 de test, con 42 turnos) y 4 escenarios de memoria (3 de test, con 22 sondas). Las preguntas de identificador no se escribieron a mano, sino que se derivaron con código. Hay 9 preguntas con fecha, y en test hay 4 de sí/no con respuesta "sí" y 4 con respuesta "no", así que un modelo que contestara siempre lo mismo no sacaría ventaja. Todo el conjunto se congeló antes de cualquier resultado (`eval/FROZEN.json`, clave `99676007c42e6f6e`).

### Preguntas de la Etapa 2

**¿Cómo se evitó que las preguntas copiaran al corpus, y qué dice el chequeo de contaminación?**

Escribimos las preguntas a partir de una lista de temas, y anotamos la respuesta esperada antes de buscarla en el corpus. Las claves de respuesta se copiaron del texto gold, pero las preguntas no, y las de identificador se derivaron con código. La idea era que la pregunta no heredara el vocabulario de la ley.

El chequeo de contaminación usa la misma normalización que el canal léxico, y dio limpio: ninguno de los 141 textos chequeados (preguntas, turnos, reescrituras oráculo y sondas) comparte 5 términos seguidos con su gold, y el máximo es 4. Además, BM25 no pone primera ninguna sección gold (sección@1 = 0 en todas las categorías), o sea que ninguna pregunta se puede resolver por coincidencia literal. Esto tiene una contracara que se ve en la Etapa 3: si las preguntas no comparten vocabulario con la ley, un recuperador léxico arranca en desventaja.

**¿Qué fracción del gold son conjuntos en lugar de ids únicos, y por qué?**

El 7% (7 de 97). En todos los casos es porque la respuesta está en dos lugares, ya sea una ley estatal y una autonómica, o dos artículos distintos. Ninguno es un conjunto por gemelos. Las secciones que solo remiten a otra norma no entran en el gold, porque no contienen la respuesta.

**¿Qué habilidades de memoria cubren los escenarios, con cuántas sondas, y cuáles quedaron afuera?**

Las seis habilidades están cubiertas en las 22 sondas de test: 6 de actualización de conocimiento, 4 multi-sesión, y 3 de cada una de las demás (extracción, temporal, abstención y preferencia). Quedaron afuera las preferencias que cambian, los datos que dijo el propio asistente y el olvido a pedido.

**Una pregunta que haría un usuario real y que el conjunto no puede medir.**

"¿Me conviene más alquilar o comprar con la ayuda del Plan Estatal?". Es una pregunta muy natural para alguien que está decidiendo dónde vivir, pero pide un consejo y una cuenta con los datos del usuario, no una regla que esté escrita en un artículo. La contención de una clave no puede puntuarla, porque no hay una frase correcta que buscar en la respuesta.

**Si `topical` e `identifier` no coinciden sobre una técnica, ¿en cuál se confía, y por qué decidirlo antes?**

En `topical`, y lo decidimos antes de ver cualquier resultado. `topical` es lo que pregunta un usuario real, mientras que `identifier` mide la búsqueda literal de un artículo por su número, que depende mucho del prefijo (el número de la norma está en el título del documento, no en la sección). Decidirlo antes evita una trampa en la que es fácil caer: elegir, después de ver la tabla, la columna que le da la razón a la técnica que uno prefería.

## 4 · Etapa 3 · Recuperación

Armamos ocho filas acumulativas. Cada una agrega una sola pieza a la anterior, y todas entregan el mismo presupuesto de 3.000 tokens del generador, de modo que la diferencia entre dos filas se pueda atribuir a esa pieza y no a que una entrega más texto que otra. La métrica primaria, pre-registrada, es **"clave entregada desde el gold"**: que al generador le haya llegado un pasaje del gold que contiene la clave. La tabla es sobre las 26 preguntas de test que tienen respuesta.

| Fila | doc@10 | sección@10 | clave desde gold | pasajes | ms (mediana) | contra la anterior (signos) |
|---|---|---|---|---|---|---|
| bm25 | 22 | 4 | 4 | 10,2 | 7 | — |
| dense | 23 | 9 | 10 | 10,8 | 16 | 7-1, p = 0,070 (dos flags) |
| +prefix | 26 | 17 | 16 | 10,7 | 15 | 7-1, p = 0,070 |
| +rrf | 26 | 18 | 16 | 11,2 | 31 | 4-4, p = 1,00 |
| +rerank | 26 | 20 | 17 | 10,0 | 224 | 4-3, p = 1,00 |
| +parent | 26 | 20 | 17 | 4,9 | 226 | 3-3, p = 1,00 |
| +date | 25 | 22 | 19 | 5,0 | 230 | 5-3, p = 0,73 |
| +region | 25 | 22 | 19 | 5,0 | 235 | 0-0 |

**Qué piezas pagaron su costo.** Ninguna pieza supera p = 0,05. La que más se acerca es el prefijo contextual (anteponerle al hijo el título, la sección y la vigencia), con 7 victorias y 1 derrota contra la fila anterior. RRF, el reranker, los padres y los filtros quedan todos dentro del ruido. Esto deja una situación incómoda, porque `+date` tiene la mayor cantidad de aciertos (19) pero no se distingue estadísticamente de `+prefix` (16). Para no elegir mirando la tabla, habíamos fijado de antemano una regla: entre las filas que no se distinguen de la de más aciertos, gana la de menos piezas. Por esa regla, **la mejor configuración es `+prefix`**, y es la que usan las Etapas 4 a 6; RRF, el reranker, los padres y los filtros quedan afuera. Puede ser que con más preguntas el filtro de fecha se ganara su lugar, pero con 26 no tenemos cómo afirmarlo, y la regla está justamente para no hacerlo. El reranker, además, multiplica por siete la latencia (de 31 a 224 ms de mediana) a cambio de una sola pregunta más.

**El movimiento más grande, con la caja abierta.** De BM25 a denso cambian 8 preguntas, 7 a favor del denso. En 4 de ellas BM25 no tiene la sección gold ni siquiera en su top 100. El usuario pregunta si puede "dejar el contrato" o quién arregla "la caldera", y la ley habla de "desistimiento" y de "reparaciones". BM25 termina en preámbulos de reales decretos-ley y en la Ley de Propiedad Horizontal, atraído por términos genéricos como "piso" o "alquiler", mientras que el denso encuentra los artículos de la LAU. **Gana por la razón que da su paper**, el desajuste de vocabulario entre la pregunta y el texto, aunque no porque ponga el gold primero: el gold queda entre los puestos 1 y 15 del denso y llega gracias a la entrega de 3.000 tokens. La única pregunta que pierde (md04) BM25 la pone segunda y el denso, 21.ª. Este resultado conecta con el chequeo de contaminación: como escribimos las preguntas sin copiar el vocabulario de la ley, el canal léxico quedó en desventaja desde el diseño. Nos parece una desventaja realista, porque un inquilino no pregunta con las palabras del legislador.

**Documento contra sección.** La brecha es grande. Con `+prefix`, el documento correcto llega en 26 de 26 preguntas, pero la sección correcta solo en 17 y la clave desde el gold en 16. El recuperador casi siempre encuentra la ley correcta, pero muchas veces no el artículo (o la versión) correcto. Para la Etapa 4 esto predecía un techo de unas 16 preguntas para la generación con recuperación, y efectivamente la fila de recuperación respondió bien 15 de 26 (sección 5).

**Hijos contra padres, con tokens igualados.** Empatan: 17 contra 17, con 3-3 en los desacuerdos. La diferencia es que los padres entregan la mitad de pasajes (4,9 contra 10,0), es decir, menos fragmentos pero más largos, y el resultado es el mismo.

**Los filtros de metadatos.** El filtro de fecha mueve 5-3 a favor (en `topical` de 6 a 8 y en `metadata` de 5 a 6), sin superar el ruido. El de región no cambia nada.

**¿Puede la recuperación sola decir que la respuesta no está?** No. El coseno máximo de las preguntas fuera del corpus (media 0,67) no es menor que el de las que tienen respuesta (0,64), y con el puntaje del reranker pasa lo mismo. En este corpus tiene sentido: una pregunta sobre la ley catalana o sobre la LEC habla de vivienda y alquiler igual que las demás, y encuentra pasajes parecidos aunque no tengan la respuesta. La decisión de abstenerse tiene que quedar del lado del generador.

Por último, verificamos que el reranker no estuviera truncando: 0 de 2.000 pares (consulta + hijo) pasan los 512 tokens de su ventana.

## 5 · Etapa 4 · Respuestas fundamentadas

Corrimos tres brazos sobre las mismas 30 preguntas, en orden fijo y con el modelo descargado al empezar cada uno.

| Brazo | Acierto | Con respuesta | Abstención correcta | Citó sin recibir |
|---|---|---|---|---|
| Libro cerrado | 5/30 | 3/26 | 2/4 | — |
| Recuperación (`+prefix`) | 18/30 | 15/26 | 3/4 | 0 |
| Oráculo (secciones gold) | 25/30 | 22/26 | 3/4 | 0 |

A diferencia de la Etapa 3, acá las dos brechas superan el ruido. Recuperación contra libro cerrado queda 14-1 (p = 0,001; +43 puntos, IC de +23 a +63), y oráculo contra recuperación, 7-0 (p = 0,016; +23 puntos, IC de +10 a +40). La segunda hay que leerla con cuidado. El oráculo recibe muchos menos tokens que la recuperación (1.086 contra 2.968), así que su ventaja mezcla dos efectos que no podemos separar: tener el pasaje correcto y tener menos distractores. Libro cerrado, por su parte, acierta 3 de 26 preguntas con respuesta, lejos del umbral de §7.4, lo que confirma que el eval set mide recuperación y no lo que el modelo aprendió en el preentrenamiento.

La tabla 2×2 separa las dos superficies de falla para la fila de recuperación, sobre las 26 preguntas con respuesta:

| | respondió bien | respondió mal |
|---|---|---|
| **recuperó bien** (clave desde el gold) | 13 | 3 |
| **recuperó mal** | 2 | 8: 4 fieles al pasaje equivocado, 3 abstenciones, 1 sin apoyarse en lo entregado |

Lo primero que muestra es que la mayoría de los errores son de recuperación y no de generación: de las 11 preguntas que se respondieron mal, 8 son casos de "no llegó", en los que el pasaje correcto nunca le llegó al generador. En esos casos el modelo se comportó de forma bastante razonable, porque 4 respuestas son fieles al pasaje equivocado que recibió, 3 son abstenciones y solo 1 no se apoya en lo que se le entregó.

Más interesantes son las 3 fallas con el pasaje correcto a la vista ("llegó y no se usó"). Dos de ellas (md02 y md05) son de razonamiento temporal. Llegaron dos versiones del mismo artículo, y el modelo aplicó la siguiente a una fecha anterior a su vigencia; por ejemplo, para un contrato de febrero de 2019 usó el texto "vigente desde marzo de 2019". La tercera (tp05) es correcta en sustancia, pero no usa la palabra de la clave, así que es más una falla de la medición que del modelo. Las dos fallas temporales nos parecen de lo más relevante del proyecto, porque son justamente el caso para el que elegimos este corpus, y muestran que recuperar la versión correcta no alcanza si el generador no razona bien sobre fechas.

En cuanto a las citas, ningún brazo citó un bloque que no se le hubiera entregado (0 en 60 respuestas con pasajes). Y con la respuesta en el contexto, el modelo se abstuvo 0 veces en recuperación y 1 en el oráculo, así que no vemos un problema de exceso de cautela.

Para mostrar un prompt demasiado grande armamos uno de 11.239 tokens con `num_ctx` de 8.192. Con la guarda activada, el ensamblador levanta `ContextOverflow` antes de mandarlo. Sin la guarda, el servidor evaluó solo 4.098 tokens (la mitad de la ventana: los primeros tokens y la cola) y el modelo respondió con total seguridad, pero sin la clave. Nada en la respuesta indicaba que se había perdido la mayor parte del contexto, y lo único que lo delató fue comparar nuestra cuenta con `prompt_eval_count`. Es el ejemplo más claro de por qué valía la pena verificar los presupuestos contra el servidor.

Para medir la varianza repetimos 10 preguntas en frío, en otro orden y sin caché. El texto cambió en 7 casos y el puntaje en ninguno.

El experimento de posición del pasaje no lo hicimos. Lo recortamos porque, a esta escala, un resultado nulo no habría refutado *Lost in the Middle*: con 30 preguntas no tendríamos poder estadístico para detectar el efecto aunque existiera.

## 6 · Etapa 5 · La conversación

### Parte A · Con qué buscar

En la Parte A comparamos cuatro formas de armar la consulta en cada turno. Son 37 turnos con respuesta en 8 guiones de test, y el ruido se mide por guion, porque los turnos de un mismo guion no son independientes.

| Consulta | doc@5 | sección@5 | clave desde gold (1.350 tokens) |
|---|---|---|---|
| último turno | 26 | 15 | 17 |
| conversación completa (el turno más nuevo primero) | 32 | 11 | 13 |
| reescritura del modelo | 26 | 16 | 16 |
| reescritura oráculo | 34 | 22 | 18 |

Por dependencia (doc@5 / sección@5):

| Dependencia | último turno | conversación | reescritura | oráculo |
|---|---|---|---|---|
| standalone (8) | 7 / 4 | 7 / 4 | 7 / 4 | 7 / 5 |
| coreference (10) | 5 / 2 | 9 / 3 | 5 / 1 | 9 / 4 |
| ellipsis (5) | 1 / 0 | 5 / 1 | 3 / 3 | 5 / 2 |
| topic_shift (6) | 6 / 5 | 4 / 0 | 3 / 3 | 6 / 6 |
| return (8) | 7 / 4 | 7 / 3 | 8 / 5 | 7 / 5 |

Por número de turno (doc@5 / sección@5):

| Turno | último turno | conversación | reescritura | oráculo |
|---|---|---|---|---|
| 1 (8) | 7 / 4 | 7 / 4 | 7 / 4 | 7 / 5 |
| 2 (8) | 2 / 1 | 7 / 3 | 4 / 2 | 8 / 4 |
| 3 (7) | 5 / 3 | 7 / 0 | 4 / 3 | 7 / 4 |
| 4 (6) | 6 / 3 | 3 / 0 | 3 / 1 | 6 / 5 |
| 5 (5) | 3 / 2 | 5 / 2 | 5 / 4 | 3 / 2 |
| 6 (3) | 3 / 2 | 3 / 2 | 3 / 2 | 3 / 2 |

Lo primero es que ninguna consulta supera el ruido. La mayor diferencia por guion es la del oráculo contra el último turno, 4-0, con p = 0,125, así que todo lo que sigue hay que leerlo como tendencias.

Aun así, las tendencias son coherentes. A nivel de documento, el ranking cambia con el turno. En los turnos 2 y 3 la conversación completa encuentra la norma (7/8 y 7/7) y el último turno no (2/8 en el turno 2), pero en el turno 4 se invierte (último turno 6/6, conversación 3/6). A nivel de sección, la conversación completa es la peor desde el turno 3 (0/7 y 0/6). Con 3 a 8 turnos por fila, esto es una dirección y no evidencia.

La explicación que encontramos es que la conversación completa encuentra la norma de la que se viene hablando, pero no el artículo nuevo que se pregunta. Por eso gana a nivel de documento en las correferencias y las elipsis, donde el turno solo no dice de qué se habla, y pierde en los cambios de tema (0/6 a sección@5), donde el historial arrastra la búsqueda hacia el tema anterior. El último turno, en cambio, falla donde uno esperaría: en el turno 2, que es la primera correferencia (2/8 a doc@5). Cada consulta tiene entonces su punto débil, y ninguna domina en todos los tipos de dependencia.

La reescritura oráculo no resultó mucho mejor que las del modelo (18 contra 16-17), como señalan Ye et al. Por guion, contra la reescritura del modelo, queda 4-1 (p = 0,375), así que no hay una diferencia que defender. Mirando por dependencia (sección@5), el oráculo le gana en los cambios de tema (6/6 contra 3/6) y en las correferencias (4/10 contra 1/10), pierde en las elipsis (2/5 contra 3/5) y empata en `return` (5/8). Por turno, pierde en el 5 (2/5 contra 4/5).

Leímos 20 reescrituras del modelo, y 15 resuelven bien la referencia. De las otras cinco, una agrega contenido de la ley que nadie dijo ("los doce meses del artículo 9 bis"), que es el efecto HyDE; otra mezcla mal el historial; una arrastra el tema anterior a un cambio de tema; una no resuelve nada, y una pierde la fecha que decide la respuesta. Esta última es la más grave para nuestro corpus, porque en una pregunta temporal la fecha es justamente lo que define qué versión hay que recuperar.

### Parte B · Qué conservar

En la Parte B la pregunta es qué conservar del historial. Cada guion se reproduce solo, con su historial guionado. El total de pasajes más historial queda fijo en 1.500 tokens y el historial tiene un tope; lo que el historial no usa pasa a los pasajes, de modo que el total es el mismo en todos los brazos (la diferencia de `prompt_eval_count` entre brazos es de 46 tokens en promedio). Como consulta usamos el último turno, por la regla pre-registrada.

| Brazo | Pierde el turno 1 (de 34) | Acierto | Clave desde gold | Llamadas por turno |
|---|---|---|---|---|
| ventana, tope 100 | 26 | 18/42 | 16/37 | 1,0 |
| ventana, tope 150 | 19 | 17/41 | 17/37 | 1,0 |
| ventana, tope 250 | 7 | 20/41 | 17/37 | 1,0 |
| resumen (MemGPT), tope 150 | — | 18/42 | 17/37 | 1,6 |
| buffer completo (referencia) | 0 | 19/41 | 17/37 | 1,0 |

El tope restringe de verdad (con 100 tokens se pierde el turno 1 en 26 de 34 casos), pero el acierto casi no se mueve. La mejor división fue la de 250 tokens de historial y el resto de los 1.500 para pasajes (20/41), aunque ninguna comparación supera el ruido: la mayor, 250 contra 150, es 3-0 por guion, con p = 0,25.

El dato que más nos dice es el del turno 2, que da 1/8 en todos los brazos. Con el último turno como consulta, el pasaje que la correferencia necesita no llega, y conservar el turno 1 en el historial no lo trae. El historial le puede servir al generador para entender la pregunta, pero no le sirve al recuperador si la consulta no lo usa. **A esta escala, con qué se busca pesa más que qué historial se guarda**, lo que conecta directamente con la Parte A.

La sonda del resumen, una pregunta derivada del turno 1 que se hace sin pasajes, sí muestra el mecanismo. Con el mismo tope de 150, la ventana recuerda el turno 1 en 1 de 7 guiones y el resumen en 3 de 7, sin tenerlo a la vista. Con el buffer completo recuerda 7 de 7, y sin historial, 0 de 7. Con 7 guiones esto no es evidencia (p = 0,625), pero va en la dirección que uno esperaría de un resumen al estilo MemGPT: conserva algo de lo que la ventana ya descartó.

Auditamos 25 actualizaciones del resumen con las categorías de Wang et al.: 0 hechos inventados, 1 relación equivocada, 16 con detalles faltantes, 2 con texto corrupto (tokens en chino) y 6 correctas. El problema de nuestro resumen no es que invente, sino que pierde. Wang et al. encuentran los tres tipos de error en proporciones parecidas; en nuestro caso el faltante domina claramente, aunque hay que tener en cuenta que ellos cuentan proporción de contenido y nosotros actualizaciones, así que los números no son directamente comparables. Los detalles faltantes tienen dos causas. Una es que el resumen **se congela en su tope de 220 caracteres** (30 de 58 actualizaciones lo tocaron) y empieza a repetir el mismo texto cortado. La otra es que **reemplaza en lugar de fusionar** y pierde lo anterior: en dl01 se perdió la fecha de 2016. Vale aclarar que lo que probamos es este resumen, con este tope.

En costo, el resumen pide 1,6 llamadas por turno contra 1,0 de la ventana, y 247 tokens de salida contra 214. A cambio da 1 acierto más que la ventana con el mismo tope (18/42 contra 17/41) y recuerda más en la sonda (3 de 7 contra 1 de 7), las dos cosas dentro del ruido. Nuestra conclusión es que a esta escala no paga su costo, y que ninguna estrategia gana por encima del ruido como para justificar uno.

Hubo 3 errores, los tres en el mismo turno (dl04.t4): la respuesta llegó al tope de 1.200 caracteres y, por la regla de la sección 1, se contó como error de llamada y no como respuesta.

## 7 · Etapa 6 · Memoria entre sesiones

Antes de leer los resultados hay que declarar en qué régimen estamos, porque cambia la interpretación: acá el historial entra en la ventana. Cada escenario de test tiene entre 820 y 947 tokens de historial, y quedan 3.794 tokens disponibles para él; incluso los cuatro escenarios seguidos (3.629) entran. Para ver el régimen en el que no entra impusimos un presupuesto, y `replay_600` recorta 28 de las 30 sondas. Con un presupuesto de 3.000 tokens, el historial dejaría de entrar en el cuarto escenario seguido, y con 600, en la tercera sesión del primero. Cuando eso pasa, la clave llega al espacio de memoria de la reproducción en 7 de 19 sondas en lugar de 13, y reproducción y almacén quedan empatados (10 contra 10).

Comparamos cinco brazos con el mismo prompt, los mismos pasajes y la misma pregunta. Lo único que cambia es el espacio de memoria (y, en `store_filter`, también el filtro de los pasajes). Los resultados son sobre las 22 sondas de test, en conteos:

| Habilidad | sin memoria | replay (3.000) | replay_600 | store | store + filtro |
|---|---|---|---|---|---|
| extracción (3) | 0 | 2 | 1 | 1 | 1 |
| multi-sesión (4) | 1 | 2 | 1 | 1 | 1 |
| temporal (3) | 1 | 1 | 1 | 0 | 0 |
| actualización (6) | 2 | 2 | 2 | 4 | 5 |
| abstención (3) | 3 | 2 | 3 | 3 | 2 |
| preferencia (3) | 1 | 2 | 2 | 1 | 1 |
| **total (22)** | **8** | **11** | **10** | **10** | **10** |
| tokens de memoria por sonda | 0 | 847 | 550 | 404 | 404 |
| clave en el espacio de memoria | 0/19 | 13/19 | 7/19 | 5/19 | 5/19 |

Ninguna comparación supera el ruido. Con 3 escenarios, el mejor resultado posible por escenario es 3-0, que da p = 0,25, así que esta etapa no podía producir un resultado significativo aunque hubiera una diferencia real. La dirección más clara es reproducción contra sin memoria, 5-2 por sonda.

Lo interesante aparece al mirar por habilidad, porque cada estrategia gana donde su diseño lo predice. La memoria gana en las actualizaciones de conocimiento (de 2 a 4-5 de 6), que es para lo que existe el camino de escritura: cuando un dato cambia, el almacén cierra el valor viejo, mientras que la reproducción ve los dos valores y no siempre elige el último. La reproducción, en cambio, gana en la extracción y en lo multi-sesión, donde hace falta el dato literal. Ahí el almacén encuentra las sesiones con evidencia (25 de 31), pero la clave está en su espacio en solo 5 de 19 sondas. La extracción resume, y lo que se resume se pierde. Es el mismo problema que vimos con el resumen de la Etapa 5, ahora en el camino de escritura de la memoria.

La predicción de Mem0 es que, donde la reproducción entra, el almacén gana en costo y pierde en exactitud. A nuestra escala se reproduce, dentro del ruido: el almacén usa 404 tokens de memoria por sonda contra 847, y acierta 10 contra 11. Pero el ahorro tiene un costo que la lectura no ve, que son las 66 llamadas de escritura para los 3 escenarios de test, unos 2.100 tokens por sonda. Con el mismo presupuesto (600), almacén y reproducción empatan (10 contra 10). En nuestro régimen, donde el historial entra, la ventaja del almacén es de costo y no de exactitud, salvo en las actualizaciones de conocimiento.

Como ejemplo de una actualización de punta a punta, en el escenario sc01 el dato "35 años" se guarda en la sesión 1. En la sesión 3 llega el mensaje "cumplí 36 el 8 de agosto", y la operación UPDATE cierra el registro viejo (le pone `valid_to`) y agrega el nuevo. La sonda del "antes" (15 de junio) ve 35 como vigente, y la del "después" (octubre), 36. El almacén no borra el dato anterior sino que lo cierra en el tiempo, igual que el corpus hace con las versiones de un artículo, y eso es lo que permite responder preguntas sobre el pasado.

También auditamos los 76 registros que escribió el extractor (56 en test). En test, 50 están bien, 3 mal y 3 son cosas que nunca se dijeron. En el escenario de dev la proporción es peor: 10 bien, 7 mal y 3 nunca dichos. Los errores tienen tres formas. Hay **etiquetas equivocadas** que, por la regla de un solo valor vigente por *subject*, terminan cerrando un dato verdadero; por ejemplo, la ciudad de la usuaria guardada como la de la arrendadora. Hay casos en que **la fecha de la sesión se toma como dato** ("firmó el contrato el 11 de mayo de 2026"). Y hay **no-hechos**, como "el usuario no especificó preferencia de citas", que **invalidan preferencias reales** dichas antes.

Lo preocupante es que **los chequeos automáticos no ven ninguno de estos errores**. El de lenguaje de hipótesis marca 0. El de fugas del análisis marca 7, pero 5 de esos registros dicen lo que efectivamente se dijo, porque el extractor parafrasea igual en su análisis y en el registro. Los no-hechos son las verdaderas "hipótesis guardadas como hechos", y solo se ven leyendo los registros uno por uno.

Por último, usar la memoria como filtro de la recuperación cambió los pasajes en 27 de 30 sondas, pero el resultado solo en 2: ayudó en una actualización y perjudicó una abstención. No hay ningún efecto que leer ahí.

## 8 · Etapa 7a · El juez

Para la segunda lectura usamos dos jueces sobre las 407 respuestas de test de las Etapas 4 a 6: `qwen3:8b`, el mismo modelo que generó las respuestas, y `granite3.3:8b`, de otra familia. Tener dos jueces nos permite medir la autopreferencia en lugar de suponerla. Los dos reciben la pregunta, la clave y la respuesta, sin el contexto ni el análisis del generador. Las abstenciones en preguntas con respuesta se cuentan como incorrectas por regla, sin llamar al juez.

Comparados con la contención, los dos jueces se van en direcciones opuestas. `qwen3:8b` es **más severo**: en las preguntas con respuesta acepta 58 de 216, contra 90 de la contención (33 contra 1 en los desacuerdos). `granite3.3:8b` es **más indulgente**: a libro cerrado acepta 14 de 30 respuestas, cuando la contención acepta 5. Entre ellos coinciden en 305 de 407 juicios, y granite acepta 97 respuestas que qwen rechaza, contra solo 5 al revés. Lo tranquilizador es que ninguno da vuelta el orden de los brazos en ninguna etapa, así que las conclusiones comparativas no dependen del instrumento.

Para saber en cuál confiar auditamos 82 juicios (el 20%), muestreados con semilla y con los desacuerdos sobrerrepresentados, sin ver los veredictos de los instrumentos:

| Instrumento contra el auditor | Acuerdo en la muestra | Reponderado | Acepta lo que el auditor rechaza | Rechaza lo que el auditor acepta |
|---|---|---|---|---|
| contención | 73/82 | 0,90 | 0 | 9 |
| `granite3.3:8b` | 69/82 | 0,86 | 8 | 5 |
| `qwen3:8b` | 42/82 | 0,75 | 4 | 36 |

La contención resultó el instrumento más cercano al auditor, y además se equivoca siempre para el mismo lado. Nunca acepta una respuesta incorrecta, y sus 9 desacuerdos son respuestas correctas dichas con otras palabras ("al día en el pago" en lugar de "al corriente"). Esto significa que los números de las Etapas 4 a 6 son una cota inferior del acierto real, lo que nos parece preferible a lo contrario.

La autopreferencia que esperábamos no aparece: el juez de la misma familia que el generador es el más severo, no el más indulgente. Lo sabemos porque lo comparamos con un segundo juez de otra familia sobre las mismas respuestas, y con la auditoría. `qwen3:8b` rechaza 36 respuestas que el auditor acepta y acepta solo 4 que el auditor rechaza. Su problema no es favorecer sus propias respuestas, sino rechazar respuestas correctas que no repiten la clave literalmente.

Nuestra conclusión es que ningún juez de 8B mejora a la contención como vara de medir para este eval set. Para respuestas abiertas, donde la contención no alcanza, el de otra familia está más cerca del auditor que el de la misma. La parte 7b, reimplementar el loop en un framework, no la hicimos (sección 10), así que no tenemos una comparación de lo que un framework decide por su cuenta.

## 9 · Lo que encontró el proyecto, más allá de las tablas

Más allá de los números de cada etapa, hay algunas cosas que aprendimos y que no entran en ninguna tabla.

La primera es que un corpus legal se mueve más rápido que un trabajo práctico. En ocho días cambiaron 6 de las 177 normas, y una foto ya traía una versión con vigencia futura que cambió el gold de una pregunta al mover el día de referencia. Fijar la foto por hash y verificar el eval set al cargarlo puede parecer burocracia, pero fue lo que detectó el problema. Sin esa verificación, una recuperación correcta se habría contado como un error.

La segunda es que el modelo lee mal la vigencia. Con el pasaje correcto y la versión siguiente del mismo artículo a la vista, aplica la versión equivocada a una fecha. Esto le pone un límite a lo que puede resolver la recuperación: el filtro de fecha puede asegurar que llegue la versión correcta, pero no alcanza si el generador no razona sobre intervalos. Creemos que es la falla más característica de este corpus, y sospechamos que aparecería en cualquier corpus con versiones.

La tercera es que, en la conversación, el cuello de botella está en buscar bien el turno dependiente. Guardar más historial no recuperó lo que la consulta no encontró. Si tuviéramos más tiempo para la Etapa 5, lo pondríamos en la consulta y no en la memoria de la sesión.

La cuarta es que una memoria con "un valor vigente por *subject*" amplifica los errores del extractor. Un no-hecho o una etiqueta equivocada no solo agregan ruido, sino que invalidan un dato verdadero, porque la regla cierra el valor anterior. Y esos errores no los detecta ningún chequeo automático; hay que leer los registros. Es un caso en el que una decisión de diseño razonable, mantener un solo valor vigente para resolver las actualizaciones, se vuelve un riesgo cuando la fuente que escribe en la memoria no es confiable.

La quinta es que los jueces de 8B no se pueden usar sin calibrar. Los dos difieren entre sí más que cualquier brazo de cualquier etapa, y el de la misma familia que el generador resultó el más severo, no el más indulgente (sección 8). La contención, que es la vara más simple, fue la que más coincidió con el auditor. Antes de reemplazar una métrica simple por un juez, hay que medir si el juez es mejor.

Por último, temperatura 0 no es determinista, pero a esta escala cambia la redacción y no el veredicto: cambiaron 7 de 10 textos y 0 de 10 puntajes.

## 10 · Prioridades, recortes y uso de IA

**El plan contra lo que pasó.** El plan inicial era hacer las ocho etapas y recortar desde el final si hacía falta. Terminamos recortando tres cosas: la Etapa 7b (reimplementar el loop en un framework), el experimento de posición del pasaje de la Etapa 4 y el de §2.5. Las razones están en el registro de decisiones. El filtro por memoria de la Etapa 6 tuvo un recorrido distinto: primero lo recortamos, y después lo hicimos, cuando la entrega se movió una semana y el filtro de la Etapa 3 ya existía.

Hubo además dos cosas que no estaban en el plan. La foto se perdió y tuvimos que rehacer las Etapas 0 a 2, con una edición del eval set. Y todas las corridas se hicieron en una sola máquina local (RTX 5070, Ollama 0.34.4, la misma versión que Colab). El presupuesto real terminó siendo menor que el estimado: unas 1.420 llamadas al LLM y ~1,5 h de GPU para las Etapas 3 a 7, contra ~7-8 h de T4 estimadas.

**Uso de IA (§2.3).** Usamos asistentes de código (Claude Code y Codex) para escribir el código de las etapas y sus tests, para revisarlo en busca de las trampas del enunciado antes de correr, y para ejecutar las corridas. Algunas decisiones de diseño también las discutimos con el asistente, pero las tomó el grupo: el corpus y el plan, los recortes, correr todo en una máquina local y rehacer la foto.
