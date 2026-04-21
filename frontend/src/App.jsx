import { useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_MATRIX = [
  [1.2, 0.4],
  [-0.3, 1.1],
];

const DEFAULT_B = [2, 1];

const OPERATIONS = [
  { value: "all", label: "全链路分析" },
  { value: "determinant", label: "行列式" },
  { value: "inverse", label: "逆矩阵" },
  { value: "eigen", label: "特征值" },
  { value: "rank", label: "矩阵秩" },
  { value: "solve", label: "求解 Ax = b" },
];

const VALUE_PROPS = [
  {
    title: "任何人都能打开",
    body: "它的目标已经不是本地演示，而是一个发链接就能访问的公开网站。",
  },
  {
    title: "像数学工具一样工作",
    body: "输入矩阵、拖动点、观察图形、阅读解释，整个过程都在同一个工作台里完成。",
  },
  {
    title: "兼顾产品感与答辩感",
    body: "既适合公开访问，也适合比赛答辩时切到更聚焦的展示模式。",
  },
];

const ROADMAP = [
  "今天先把网站骨架、工作台布局和部署底座打稳。",
  "接下来补案例详情、帮助引导、对外发布说明和公开访问叙事。",
  "最后两天集中做云部署、答辩模式、移动端和最后收口。",
];

const OPERATIONS_SET = new Set(OPERATIONS.map((item) => item.value));

function parseMatrixParam(value) {
  if (!value) return null;
  const rows = value
    .split(";")
    .map((row) => row.split(",").map((cell) => Number(cell)));
  if (
    rows.length !== 2 ||
    rows.some(
      (row) => row.length !== 2 || row.some((cell) => Number.isNaN(cell)),
    )
  ) {
    return null;
  }
  return rows;
}

function parseVectorParam(value) {
  if (!value) return null;
  const items = value.split(",").map((cell) => Number(cell));
  if (items.length !== 2 || items.some((cell) => Number.isNaN(cell))) {
    return null;
  }
  return items;
}

function readInitialState() {
  if (typeof window === "undefined") {
    return {
      matrix: DEFAULT_MATRIX,
      bVector: DEFAULT_B,
      operation: "all",
      caseId: "",
      view: "public",
    };
  }

  const params = new URLSearchParams(window.location.search);
  const matrix = parseMatrixParam(params.get("m")) ?? DEFAULT_MATRIX;
  const bVector = parseVectorParam(params.get("b")) ?? DEFAULT_B;
  const operationParam = params.get("op");
  const operation = OPERATIONS_SET.has(operationParam) ? operationParam : "all";
  const caseId = params.get("case") ?? "";
  const view = params.get("view") === "defense" ? "defense" : "public";

  return { matrix, bVector, operation, caseId, view };
}

function formatMatrixParam(matrix) {
  return matrix
    .map((row) => row.map((value) => Number(value).toFixed(2)).join(","))
    .join(";");
}

function formatVectorParam(vector) {
  return vector.map((value) => Number(value).toFixed(2)).join(",");
}

function multiplyMatrixVector(matrix, point) {
  return [
    matrix[0][0] * point[0] + matrix[0][1] * point[1],
    matrix[1][0] * point[0] + matrix[1][1] * point[1],
  ];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function roundNumber(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function solutionLabel(solution) {
  if (!solution) return "等待分析";
  return solution.type_label || solution.warning || "等待分析";
}

function geometryNarrative(determinantEstimate) {
  if (Math.abs(determinantEstimate) < 0.1) {
    return "面积几乎塌缩，说明这个变换已经接近把平面压扁。";
  }
  if (determinantEstimate < 0) {
    return "方向发生翻转，说明这个变换带有镜像性质。";
  }
  return "方向保持不变，主要体现拉伸、旋转或剪切。";
}

function App() {
  const initialState = useMemo(() => readInitialState(), []);
  const [matrix, setMatrix] = useState(initialState.matrix);
  const [bVector, setBVector] = useState(initialState.bVector);
  const [operation, setOperation] = useState(initialState.operation);
  const [activeCaseId, setActiveCaseId] = useState(initialState.caseId);
  const [defenseMode, setDefenseMode] = useState(initialState.view === "defense");
  const [probePoint, setProbePoint] = useState([1.35, 0.95]);
  const [cases, setCases] = useState([]);
  const [siteContent, setSiteContent] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [shareMessage, setShareMessage] = useState("");

  const roundedMatrix = useMemo(
    () => matrix.map((row) => row.map((value) => Number(value))),
    [matrix],
  );

  const determinantEstimate = useMemo(
    () =>
      roundedMatrix[0][0] * roundedMatrix[1][1] -
      roundedMatrix[0][1] * roundedMatrix[1][0],
    [roundedMatrix],
  );

  const transformedProbe = useMemo(
    () => multiplyMatrixVector(roundedMatrix, probePoint),
    [roundedMatrix, probePoint],
  );

  const activeCase = useMemo(
    () => cases.find((item) => item.id === activeCaseId) ?? null,
    [cases, activeCaseId],
  );

  const fetchAnalysis = async (
    nextOperation = operation,
    nextMatrix = roundedMatrix,
    nextB = bVector,
  ) => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          matrix: nextMatrix,
          operation: nextOperation,
          b: nextB,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "分析请求失败");
      }
      setAnalysis(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const boot = async () => {
      try {
        const [contentResponse, casesResponse] = await Promise.all([
          fetch("/api/site-content"),
          fetch("/api/demo-cases"),
        ]);
        const contentData = await contentResponse.json();
        const casesData = await casesResponse.json();
        const fetchedCases = casesData.items || [];
        setSiteContent(contentData);
        setCases(fetchedCases);

        const matchedCase = fetchedCases.find(
          (item) => item.id === initialState.caseId,
        );
        if (matchedCase) {
          setMatrix(matchedCase.matrix);
          setBVector(matchedCase.b ?? DEFAULT_B);
          setOperation(matchedCase.operation ?? "all");
          setActiveCaseId(matchedCase.id);
          fetchAnalysis(
            matchedCase.operation ?? "all",
            matchedCase.matrix,
            matchedCase.b ?? DEFAULT_B,
          );
          return;
        }
      } catch {
        setCases([]);
      }

      fetchAnalysis(
        initialState.operation,
        initialState.matrix,
        initialState.bVector,
      );
    };

    boot();
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set("m", formatMatrixParam(matrix));
    params.set("b", formatVectorParam(bVector));
    params.set("op", operation);
    if (activeCaseId) {
      params.set("case", activeCaseId);
    } else {
      params.delete("case");
    }
    if (defenseMode) {
      params.set("view", "defense");
    } else {
      params.delete("view");
    }
    const nextUrl = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
    window.history.replaceState({}, "", nextUrl);
  }, [matrix, bVector, operation, activeCaseId, defenseMode]);

  const summaryCards = useMemo(() => {
    const results = analysis?.results ?? {};
    return [
      {
        label: "det(A)",
        value: results.determinant?.display ?? roundNumber(determinantEstimate),
      },
      {
        label: "面积缩放",
        value: `${roundNumber(determinantEstimate)}x`,
      },
      {
        label: "方程组",
        value: solutionLabel(results.solution),
      },
      {
        label: "矩阵秩",
        value:
          results.rank?.rank !== undefined
            ? `${results.rank.rank}`
            : analysis?.diagnostics?.rank_estimate ?? "2",
      },
    ];
  }, [analysis, determinantEstimate]);

  const steps = useMemo(() => {
    const results = analysis?.results;
    if (!results) return [];

    if (results.solution?.steps?.length) {
      return results.solution.steps.slice(0, 6);
    }
    if (results.inverse?.steps?.length) {
      return results.inverse.steps.slice(0, 6);
    }
    if (results.determinant?.steps?.length) {
      return results.determinant.steps.slice(0, 6);
    }
    return [];
  }, [analysis]);

  const applyCase = (item) => {
    setMatrix(item.matrix);
    setBVector(item.b ?? DEFAULT_B);
    setOperation(item.operation ?? "all");
    setActiveCaseId(item.id);
    setProbePoint([1.2, 0.8]);
    fetchAnalysis(item.operation ?? "all", item.matrix, item.b ?? DEFAULT_B);
  };

  const resetWorkspace = () => {
    setMatrix(DEFAULT_MATRIX);
    setBVector(DEFAULT_B);
    setOperation("all");
    setActiveCaseId("");
    setProbePoint([1.35, 0.95]);
    fetchAnalysis("all", DEFAULT_MATRIX, DEFAULT_B);
  };

  const copyShareLink = async () => {
    const shareUrl = window.location.href;
    try {
      await navigator.clipboard.writeText(shareUrl);
      setShareMessage(
        defenseMode
          ? "已复制当前答辩视图链接，别人打开后会直接进入同样的展示模式。"
          : "已复制当前工作台链接，别人打开后会看到相同的矩阵状态。",
      );
    } catch {
      setShareMessage(`复制失败，请手动复制这个链接：${shareUrl}`);
    }
  };

  const diagnostics = analysis?.diagnostics ?? {};
  const results = analysis?.results ?? {};
  const sections = siteContent?.sections ?? [];
  const quickstart = siteContent?.quickstart ?? [];
  const faq = siteContent?.faq ?? [];
  const deployment = siteContent?.deployment ?? [];

  return (
    <div className={`app-shell ${defenseMode ? "defense-mode" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">M</div>
          <div>
            <div className="eyebrow">MatrixVis Public Web App</div>
            <strong>MatrixVis</strong>
          </div>
        </div>
        <nav className="nav-links">
          <a href="#workspace">实验室</a>
          <a href="#cases">案例</a>
          <a href="#guide">帮助</a>
          <a href="#roadmap">路线图</a>
        </nav>
        <div className="topbar-actions">
          <button
            className={`mode-toggle ${defenseMode ? "active" : ""}`}
            onClick={() => setDefenseMode((current) => !current)}
          >
            {defenseMode ? "退出答辩模式" : "切换答辩模式"}
          </button>
          <a className="launch-link" href="#workspace">
            进入工作台
          </a>
        </div>
      </header>

      <main>
        <section className="hero-section">
          <div className="hero-copy">
            <div className="eyebrow">
              {defenseMode ? "Defense View" : "Version 2 · Public Product Direction"}
            </div>
            <h1>
              {defenseMode
                ? "用更聚焦的展示视图讲清楚 MatrixVis 的矩阵故事"
                : "把 MatrixVis 做成任何人都能打开的线性代数网站"}
            </h1>
            <p>
              {defenseMode
                ? "答辩模式会弱化外围说明内容，把注意力集中到矩阵输入、图形联动和结果解释上，更适合现场展示。"
                : "这不是本地演示壳，而是一个真正面向公开访问的数学交互产品。我们保留矩阵计算能力，把体验升级成类似 GeoGebra 的图形工作台。"}
            </p>
            <div className="hero-actions">
              <a className="primary-link" href="#workspace">
                {defenseMode ? "直接进入展示工作台" : "现在体验实验室"}
              </a>
              {!defenseMode && (
                <a className="ghost-link" href="#guide">
                  看看怎么使用
                </a>
              )}
              <button className="secondary share-button" onClick={copyShareLink}>
                复制分享链接
              </button>
            </div>
            {shareMessage && <p className="share-message">{shareMessage}</p>}
          </div>

          <div className="hero-board">
            <div className="hero-board-head">
              <span>{defenseMode ? "当前展示状态" : "公开站点定位"}</span>
              <strong>
                {defenseMode
                  ? "聚焦图形联动、步骤解释和案例切换的现场演示视图"
                  : siteContent?.subtitle || "GeoGebra 风格的矩阵可视化网站"}
              </strong>
            </div>
            <div className="hero-metric-grid">
              <MetricCard
                label="站点形态"
                value={defenseMode ? "答辩展示中" : "浏览器直接访问"}
              />
              <MetricCard label="交互核心" value="图形 + 代数同步" />
              <MetricCard label="当前矩阵" value={diagnostics.shape || "2 x 2"} />
              <MetricCard
                label="探针点"
                value={`(${roundNumber(probePoint[0], 2)}, ${roundNumber(
                  probePoint[1],
                  2,
                )})`}
              />
            </div>
          </div>
        </section>

        {!defenseMode && (
          <>
            <section className="value-strip">
              {VALUE_PROPS.map((item) => (
                <article className="value-card" key={item.title}>
                  <strong>{item.title}</strong>
                  <p>{item.body}</p>
                </article>
              ))}
            </section>

            <section className="guide-section" id="guide">
              <div className="section-head">
                <div>
                  <div className="eyebrow">Quick Start</div>
                  <h2>第一次访问也能立刻上手</h2>
                </div>
                <p>
                  一个真正面向公网的网站不能只靠会演示的人来带路，所以这里把使用路径讲清楚。
                </p>
              </div>

              <div className="guide-grid">
                {quickstart.map((item, index) => (
                  <article className="guide-card" key={item.title}>
                    <span>Step {index + 1}</span>
                    <strong>{item.title}</strong>
                    <p>{item.body}</p>
                  </article>
                ))}
              </div>
            </section>
          </>
        )}

        <section className="workspace-section" id="workspace">
          <div className="section-head">
            <div>
              <div className="eyebrow">Core Workspace</div>
              <h2>{defenseMode ? "答辩展示工作台" : "GeoGebra 风格的矩阵工作台"}</h2>
            </div>
            <p>
              {defenseMode
                ? "当前视图专门为展示而优化，把主要注意力集中到左侧输入、中央图形和右侧解释。"
                : "左侧是代数输入和案例切换，中间是图形画布，右侧是结果摘要和步骤解释。这会是后续公网版本的核心界面。"}
            </p>
          </div>

          {defenseMode && (
            <div className="defense-banner">
              <strong>答辩模式已开启</strong>
              <p>
                这个视图会隐藏部分外围说明，让现场观众更快聚焦在图形联动、案例切换和结果讲解上。
              </p>
            </div>
          )}

          <div className="workspace-shell">
            <aside className="panel algebra-panel">
              <div className="panel-header">
                <div className="panel-kicker">Algebra</div>
                <h3>输入与控制</h3>
              </div>

              <label className="field-block">
                <span>矩阵 A</span>
                <div className="matrix-grid">
                  {matrix.map((row, rowIndex) =>
                    row.map((value, colIndex) => (
                      <input
                        key={`${rowIndex}-${colIndex}`}
                        type="number"
                        step="0.1"
                        value={value}
                        onChange={(event) => {
                          const next = matrix.map((sourceRow) => [...sourceRow]);
                          next[rowIndex][colIndex] = Number(event.target.value);
                          setMatrix(next);
                          setActiveCaseId("");
                        }}
                      />
                    )),
                  )}
                </div>
              </label>

              <label className="field-block">
                <span>向量 b</span>
                <div className="vector-grid">
                  {bVector.map((value, index) => (
                    <input
                      key={index}
                      type="number"
                      step="0.1"
                      value={value}
                      onChange={(event) => {
                        const next = [...bVector];
                        next[index] = Number(event.target.value);
                        setBVector(next);
                        setActiveCaseId("");
                      }}
                    />
                  ))}
                </div>
              </label>

              <label className="field-block">
                <span>分析模式</span>
                <select
                  value={operation}
                  onChange={(event) => {
                    setOperation(event.target.value);
                    setActiveCaseId("");
                  }}
                >
                  {OPERATIONS.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>

              <div className="button-row">
                <button onClick={() => fetchAnalysis()} disabled={loading}>
                  {loading ? "分析中..." : "开始分析"}
                </button>
                <button className="secondary" onClick={resetWorkspace}>
                  重置
                </button>
              </div>

              <div className="mini-note">
                <strong>{defenseMode ? "展示提醒" : "产品方向"}</strong>
                <p>
                  {defenseMode
                    ? "建议先切一个案例，再拖动左图探针点，最后带着观众读右侧结果区和步骤卡片。"
                    : "后续这里会继续扩展成案例库、历史记录、步骤回放、发布分享和课堂模式入口。"}
                </p>
              </div>

              <div className="mini-note">
                <strong>公开访问能力</strong>
                <p>
                  当前工作台状态会写入 URL 参数，便于直接分享给老师、队友或评委。
                </p>
              </div>
            </aside>

            <section className="panel graphics-panel">
              <div className="panel-header">
                <div className="panel-kicker">Graphics</div>
                <h3>图形视图</h3>
              </div>

              <div className="stage-caption">
                <strong>拖动左图橙色探针点</strong>
                <p>
                  单位方格、基向量和对应点会同步变化，让矩阵运算从符号变成视觉体验。
                </p>
              </div>

              <TransformWorkspace
                matrix={roundedMatrix}
                probePoint={probePoint}
                onProbeChange={setProbePoint}
              />

              <div className="graphics-footer">
                <article>
                  <span>原始点</span>
                  <strong>
                    ({roundNumber(probePoint[0], 2)}, {roundNumber(probePoint[1], 2)})
                  </strong>
                </article>
                <article>
                  <span>变换后</span>
                  <strong>
                    ({roundNumber(transformedProbe[0], 2)}, {roundNumber(
                      transformedProbe[1],
                      2,
                    )})
                  </strong>
                </article>
                <article>
                  <span>几何含义</span>
                  <strong>{geometryNarrative(determinantEstimate)}</strong>
                </article>
              </div>
            </section>

            <aside className="panel results-panel">
              <div className="panel-header">
                <div className="panel-kicker">Results</div>
                <h3>解释与结果</h3>
              </div>

              {error && <div className="error-box">{error}</div>}

              <div className="summary-grid">
                {summaryCards.map((card) => (
                  <article className="summary-card" key={card.label}>
                    <span>{card.label}</span>
                    <strong>{card.value}</strong>
                  </article>
                ))}
              </div>

              <InfoBlock title="运算模式">
                {OPERATIONS.find((item) => item.value === operation)?.label ||
                  "全链路分析"}
              </InfoBlock>
              <InfoBlock title="条件数">
                {diagnostics.condition_number
                  ? roundNumber(diagnostics.condition_number, 2)
                  : "待计算"}
              </InfoBlock>
              <InfoBlock title="特征值">
                {results.eigen?.values?.join(" , ") || "等待分析"}
              </InfoBlock>
              <InfoBlock title="方程组结论">
                {solutionLabel(results.solution)}
              </InfoBlock>

              <section className="steps-panel">
                <div className="steps-head">
                  <strong>步骤卡片</strong>
                  <span>Explainable Math</span>
                </div>
                {steps.length ? (
                  <div className="steps-list">
                    {steps.map((step, index) => (
                      <article className="step-card" key={`${step.description}-${index}`}>
                        <span>Step {index + 1}</span>
                        <strong>{step.description}</strong>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="muted">
                    点击“开始分析”后，这里会展示与当前运算同步的推导步骤。
                  </p>
                )}
              </section>
            </aside>
          </div>
        </section>

        <section className="cases-section" id="cases">
          <div className="section-head">
            <div>
              <div className="eyebrow">Case Gallery</div>
              <h2>{defenseMode ? "一键切换展示案例" : "一键切换教学与答辩案例"}</h2>
            </div>
            <p>
              {defenseMode
                ? "现场演示时建议从这里切入，把案例变成讲解的主线。"
                : "案例库会成为公开网站的重要入口，让第一次访问的人不用先懂参数，也能立刻看到效果。"}
            </p>
          </div>

          <div className="case-grid">
            {cases.map((item) => (
              <button
                className={`case-card ${item.id === activeCaseId ? "active" : ""}`}
                key={item.id}
                onClick={() => applyCase(item)}
              >
                <span>{item.title}</span>
                <strong>{item.subtitle}</strong>
              </button>
            ))}
          </div>

          {activeCase && (
            <article className="case-detail-card">
              <div>
                <div className="eyebrow">Active Case</div>
                <h3>{activeCase.title}</h3>
                <p>{activeCase.story}</p>
              </div>
              <div className="case-detail-grid">
                <InfoBlock title="讲解重点">{activeCase.teaching_focus}</InfoBlock>
                <InfoBlock title="推荐操作">{activeCase.action_hint}</InfoBlock>
                <InfoBlock title="推荐模式">
                  {OPERATIONS.find((item) => item.value === activeCase.operation)
                    ?.label || activeCase.operation}
                </InfoBlock>
              </div>
            </article>
          )}
        </section>

        {!defenseMode && (
          <>
            <section className="sections-strip">
              {sections.map((section) => (
                <article className="section-card" key={section.id}>
                  <span>{section.label}</span>
                  <strong>{section.description}</strong>
                </article>
              ))}
            </section>

            <section className="faq-section">
              <div className="section-head">
                <div>
                  <div className="eyebrow">FAQ</div>
                  <h2>公开网站视角下的常见问题</h2>
                </div>
                <p>
                  这些问题决定了它是一个真正可公开访问的产品，还是一个只能本地演示的页面。
                </p>
              </div>

              <div className="faq-grid">
                {faq.map((item) => (
                  <article className="faq-card" key={item.question}>
                    <strong>{item.question}</strong>
                    <p>{item.answer}</p>
                  </article>
                ))}
              </div>
            </section>

            <section className="roadmap-section" id="roadmap">
              <div className="section-head">
                <div>
                  <div className="eyebrow">4-Day Sprint</div>
                  <h2>四天内把它推向公网版</h2>
                </div>
                <p>
                  现在已经从本地演示思路切换到产品站思路。后续冲刺重点会放在部署、发布入口、移动端和更完整的教学叙事。
                </p>
              </div>

              <div className="roadmap-grid">
                {ROADMAP.map((item) => (
                  <article className="roadmap-card" key={item}>
                    <strong>{item}</strong>
                  </article>
                ))}
              </div>
            </section>

            <section className="deployment-section">
              <div className="section-head">
                <div>
                  <div className="eyebrow">Deployment Ready</div>
                  <h2>已经开始按公网部署形态组织</h2>
                </div>
                <p>
                  现在的前后端已经是同源单站点结构，后续可以直接打包成容器部署到云端，而不是依赖本地脚本演示。
                </p>
              </div>

              <div className="deployment-grid">
                {deployment.map((item) => (
                  <article className="section-card" key={item.title}>
                    <span>{item.title}</span>
                    <strong>{item.body}</strong>
                  </article>
                ))}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function InfoBlock({ title, children }) {
  return (
    <div className="info-block">
      <span>{title}</span>
      <strong>{children}</strong>
    </div>
  );
}

function TransformWorkspace({ matrix, probePoint, onProbeChange }) {
  return (
    <div className="transform-stage">
      <GraphPanel
        title="原始平面"
        mode="source"
        matrix={matrix}
        probePoint={probePoint}
        onProbeChange={onProbeChange}
      />
      <GraphPanel
        title="变换后平面"
        mode="target"
        matrix={matrix}
        probePoint={probePoint}
      />
    </div>
  );
}

function GraphPanel({ title, mode, matrix, probePoint, onProbeChange }) {
  const svgRef = useRef(null);
  const size = 340;
  const extent = 4;
  const center = size / 2;
  const unit = size / (extent * 2);
  const activePoint =
    mode === "source" ? probePoint : multiplyMatrixVector(matrix, probePoint);

  const toSvg = ([x, y]) => [center + x * unit, center - y * unit];
  const fromSvg = (clientX, clientY) => {
    const bounds = svgRef.current.getBoundingClientRect();
    const x = (clientX - bounds.left - center) / unit;
    const y = (center - (clientY - bounds.top)) / unit;
    return [clamp(x, -extent, extent), clamp(y, -extent, extent)];
  };

  const sourceBasis = [
    [1, 0],
    [0, 1],
  ];
  const targetBasis = sourceBasis.map((vector) =>
    multiplyMatrixVector(matrix, vector),
  );
  const basis = mode === "source" ? sourceBasis : targetBasis;

  const sourceSquare = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ];
  const square =
    mode === "source"
      ? sourceSquare
      : sourceSquare.map((point) => multiplyMatrixVector(matrix, point));

  const gridLines = [];
  for (let i = -extent; i <= extent; i += 1) {
    const vertical = [
      [i, -extent],
      [i, extent],
    ];
    const horizontal = [
      [-extent, i],
      [extent, i],
    ];
    [vertical, horizontal].forEach((line) => {
      const actualLine =
        mode === "source"
          ? line
          : line.map((point) => multiplyMatrixVector(matrix, point));
      gridLines.push(actualLine);
    });
  }

  const pointSvg = toSvg(activePoint);
  const squarePath =
    square
      .map((point, index) => {
        const [x, y] = toSvg(point);
        return `${index === 0 ? "M" : "L"} ${x} ${y}`;
      })
      .join(" ") + " Z";

  const startDrag = (event) => {
    if (mode !== "source" || !onProbeChange) return;
    const update = (moveEvent) => {
      onProbeChange(fromSvg(moveEvent.clientX, moveEvent.clientY));
    };
    update(event);
    window.addEventListener("pointermove", update);
    window.addEventListener(
      "pointerup",
      () => window.removeEventListener("pointermove", update),
      { once: true },
    );
  };

  return (
    <article className="graph-card">
      <div className="graph-title">{title}</div>
      <svg ref={svgRef} viewBox={`0 0 ${size} ${size}`} onPointerDown={startDrag}>
        <rect x="0" y="0" width={size} height={size} rx="26" className="graph-bg" />
        {gridLines.map((line, index) => {
          const [x1, y1] = toSvg(line[0]);
          const [x2, y2] = toSvg(line[1]);
          return (
            <line
              key={index}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              className="grid-line"
            />
          );
        })}
        <line x1={0} y1={center} x2={size} y2={center} className="axis-line" />
        <line x1={center} y1={0} x2={center} y2={size} className="axis-line" />
        <path d={squarePath} className="square-shape" />

        {basis.map((vector, index) => {
          const [x, y] = toSvg(vector);
          return (
            <g key={index}>
              <line
                x1={center}
                y1={center}
                x2={x}
                y2={y}
                className={`basis-line basis-${index}`}
              />
              <circle cx={x} cy={y} r="5" className={`basis-dot basis-${index}`} />
            </g>
          );
        })}

        <circle cx={pointSvg[0]} cy={pointSvg[1]} r="8" className="probe-dot" />
      </svg>
      <div className="graph-footer">
        {mode === "source"
          ? `P = (${roundNumber(probePoint[0], 2)}, ${roundNumber(
              probePoint[1],
              2,
            )})`
          : `T(P) = (${roundNumber(activePoint[0], 2)}, ${roundNumber(
              activePoint[1],
              2,
            )})`}
      </div>
    </article>
  );
}

export default App;
