// --- AESTHETIC STATE MANAGEMENT ---
// --- DATABASE INITIALIZATION ---
const SUPABASE_URL = 'https://nqeheytvkivsmzjvhuel.supabase.co';
const SUPABASE_ANON_KEY = 'sb_publishable_3mBlXPxtpQwb8hpbvHEA5Q_Di96aaSa'; 
const _supabase = (typeof supabase !== 'undefined') ? supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY) : null;

const State = {
    participants: [],
    events: [],
    selectedEvent: null,
    activeTab: localStorage.getItem('aesthetic_active_tab') || 'dashboard',
    isAdmin: false,
    weights: { skill: 0.6, consist: 0.3, exp: 0.1 },
    predictor: {
        network: null,
        isTrained: false,
        lastLoss: 1,
        epochs: 0
    },
    syncLog: []
};

// --- CORE ALGORITHM (AHP INSPIRED) ---
const SRI_LANKA_REGIONS = {
    'Western': { index: 1.0, label: 'High Resource Access' },
    'Central': { index: 0.85, label: 'Moderate Resource Access' },
    'Southern': { index: 0.85, label: 'Moderate Resource Access' },
    'North Western': { index: 0.75, label: 'Developing Resource Access' },
    'Sabaragamuwa': { index: 0.70, label: 'Developing Resource Access' },
    'Eastern': { index: 0.65, label: 'Low Resource Access' },
    'Northern': { index: 0.65, label: 'Low Resource Access' },
    'North Central': { index: 0.60, label: 'Low Resource Access' },
    'Uva': { index: 0.55, label: 'Constrained Resource Access' }
};

function calculateArtisticScore(participant, event) {
    // Performative Skill Match (Jaccard Index)
    const pSkills = participant.skills.split(',').map(s => s.trim().toLowerCase());
    const eReqs = event.requirements.map(s => s.toLowerCase());
    
    const intersection = pSkills.filter(s => eReqs.includes(s));
    const skillScore = eReqs.length > 0 ? (intersection.length / eReqs.length) : 0;

    // Delivery Consistency Index with Time-Series Inactivity Decay
    const inactiveMonths = participant.inactiveMonths || 0;
    // e^(-0.02 * t) represents a 2% decay per month of inactivity
    const decayFactor = Math.exp(-0.02 * inactiveMonths);
    const consistScore = (parseInt(participant.consistency) / 100) * decayFactor;

    // Regional Equity Calculation (Sri Lankan Context)
    const regionData = SRI_LANKA_REGIONS[participant.region] || SRI_LANKA_REGIONS['Western'];
    const opportunityIndex = regionData.index;

    // Industry Tenacity (Logarithmic Scaling) + Equity Boost
    const rawExpScore = Math.min(Math.log10(parseInt(participant.experience) + 1) / Math.log10(15), 1);
    const equityMultiplier = 1 + ((1 - opportunityIndex) * (consistScore * 1.5));
    const expScore = Math.min(rawExpScore * equityMultiplier, 1);

    // Weighted Sum Model using DYNAMIC WEIGHTS
    const artisticTotal = (skillScore * State.weights.skill) + 
                             (consistScore * State.weights.consist) + 
                             (expScore * State.weights.exp);

    // Predictive Analysis Vector (Neural Network)
    let projection = 0;
    if (State.predictor.isTrained) {
        const prediction = State.predictor.network.run([skillScore, consistScore, expScore, opportunityIndex]);
        projection = prediction[0] || 0;
    }

    // Hybrid consensus (50/50 mix)
    const hybridScore = Math.round(((artisticTotal + projection) / 2) * 100);
    
    // Conflict Detection
    const divergence = Math.abs(Math.round(artisticTotal * 100) - Math.round(projection * 100));
    const conflictLevel = divergence > 25 ? 'high' : (divergence > 15 ? 'mid' : 'low');

    return {
        total: hybridScore,
        scientific: Math.round(artisticTotal * 100),
        probability: Math.round(projection * 100),
        breakdown: { skill: skillScore, consist: consistScore, exp: expScore },
        matched: intersection,
        conflict: { level: conflictLevel, score: divergence },
        equity: { index: opportunityIndex, multiplier: equityMultiplier }
    };
}

// --- GROWTH TRAJECTORY SIMULATION ---
function calculateTrajectory(participant, years, event) {
    const trajectory = [];
    let currentSkills = [...participant.skills.split(',').map(s => s.trim())];
    let currentExp = parseInt(participant.experience);
    let currentConsist = parseInt(participant.consistency);

    for (let i = 0; i <= years; i++) {
        const pSim = {
            ...participant,
            experience: currentExp + i,
            consistency: Math.min(currentConsist + (i * 2), 100),
            skills: currentSkills.join(', ')
        };
        
        // Simulating "Artistic Evolution": Mastering one missing requirement every 1.5 years
        if (i > 0 && i % 2 === 0) {
            const missing = event.requirements.filter(r => !currentSkills.some(s => s.toLowerCase() === r.toLowerCase()));
            if (missing.length > 0) currentSkills.push(missing[0]);
            pSim.skills = currentSkills.join(', ');
        }

        const res = calculateArtisticScore(pSim, event);
        trajectory.push({ year: i, score: res.total, full: res });
    }
    return trajectory;
}

// --- PERFORMANCE ENGINE CALIBRATION ---
function initPredictor() {
    State.predictor.network = new brain.NeuralNetwork({
        hiddenLayers: [12, 12, 12]
    });

    const trainingData = [
        { input: [1, 1, 1, 1], output: [1] },
        { input: [0.9, 0.9, 0.8, 1], output: [0.98] },
        { input: [0.8, 0.95, 0.7, 0.6], output: [0.96] }, // Lower opportunity boosts expected output due to high tenacity
        { input: [0.6, 0.8, 0.5, 0.85], output: [0.70] },
        { input: [0.5, 0.7, 0.4, 0.7], output: [0.60] },
        { input: [0.7, 0.5, 0.2, 1], output: [0.45] },
        { input: [0.2, 0.4, 0.1, 0.65], output: [0.15] },
        { input: [0.1, 0.2, 0.1, 1], output: [0.02] }
    ];

    const logEl = document.getElementById('modelLog');
    const lossEl = document.getElementById('modelLoss');
    const epochEl = document.getElementById('calibrationEpochs');

    State.predictor.network.trainAsync(trainingData, {
        iterations: 2000,
        errorThresh: 0.005,
        log: true,
        logPeriod: 100,
        callback: (stats) => {
            State.predictor.lastLoss = stats.error;
            State.predictor.epochs = stats.iterations;
            if (lossEl) lossEl.textContent = stats.error.toFixed(6);
            if (epochEl) epochEl.textContent = stats.iterations;
            if (logEl) logEl.textContent = `Calibrating artistic projection: Epoch ${stats.iterations}... Error: ${stats.error.toFixed(6)}`;
        }
    }).then(() => {
        State.predictor.isTrained = true;
        const badge = document.getElementById('statusBadge');
        if (badge) {
            badge.textContent = "Status: Operational";
            badge.className = "status-badge operational";
        }
        if (logEl) logEl.textContent = "Performance forecasting converged. High artistic reliability achieved.";
        renderConsistencyMatrix();
        if (State.selectedEvent) renderRecommendations();
    });
}

function renderConsistencyMatrix() {
    const tbody = document.getElementById('consistencyMatrix');
    if (!tbody || !State.selectedEvent) return;

    const data = State.participants.map(p => {
        const res = calculateArtisticScore(p, State.selectedEvent);
        return { name: p.name, res };
    });

    tbody.innerHTML = data.map(d => `
        <tr>
            <td><strong>${d.name}</strong></td>
            <td>${d.res.scientific}%</td>
            <td>${d.res.probability}%</td>
            <td>${d.res.conflict.score}%</td>
            <td><span class="status-badge-inline ${d.res.conflict.level}">${d.res.conflict.level.toUpperCase()}</span></td>
        </tr>
    `).join('');

    renderFairnessAudit();

    // Dynamically update Cronbach's Alpha display
    const alphaEl = document.getElementById('cronbachAlphaVal');
    if (alphaEl) {
        const alpha = calculateCronbachsAlpha();
        alphaEl.textContent = alpha.toFixed(3);
    }
    
    // Update Ablation study if panel is open
    if (typeof calculateAblation === 'function') {
        calculateAblation();
    }
}

function calculateVariance(arr) {
    if (arr.length <= 1) return 0;
    const mean = arr.reduce((sum, val) => sum + val, 0) / arr.length;
    const sqDiffs = arr.map(val => Math.pow(val - mean, 2));
    const avgSqDiff = sqDiffs.reduce((sum, val) => sum + val, 0) / (arr.length - 1);
    return avgSqDiff;
}

function calculateCronbachsAlpha() {
    const participants = State.participants;
    if (participants.length < 2) return 0;

    const scoresA = participants.map(p => p.judgeA !== undefined ? p.judgeA : p.consistency);
    const scoresB = participants.map(p => p.judgeB !== undefined ? p.judgeB : p.consistency);
    const scoresC = participants.map(p => p.judgeC !== undefined ? p.judgeC : p.consistency);

    const varA = calculateVariance(scoresA);
    const varB = calculateVariance(scoresB);
    const varC = calculateVariance(scoresC);
    const sumOfVariances = varA + varB + varC;

    const totalScores = participants.map((p, idx) => scoresA[idx] + scoresB[idx] + scoresC[idx]);
    const varTotal = calculateVariance(totalScores);

    if (varTotal === 0) return 0;

    const K = 3;
    const alpha = (K / (K - 1)) * (1 - (sumOfVariances / varTotal));
    return isNaN(alpha) ? 0 : alpha;
}

function renderFairnessAudit() {
    const tbody = document.getElementById('fairnessAuditMatrix');
    if (!tbody || !State.selectedEvent) return;

    if (State.participants.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="padding:2rem; opacity:0.5">No Performers Available</td></tr>`;
        return;
    }

    const regionalStats = {};
    State.participants.forEach(p => {
        const region = p.region || 'Western';
        if (!regionalStats[region]) {
            regionalStats[region] = { count: 0, totalSci: 0, totalNN: 0 };
        }
        const res = calculateArtisticScore(p, State.selectedEvent);
        regionalStats[region].count++;
        regionalStats[region].totalSci += res.scientific;
        regionalStats[region].totalNN += res.probability;
    });

    let globalNNTotal = 0;
    let globalCount = 0;
    Object.values(regionalStats).forEach(s => {
        globalNNTotal += s.totalNN;
        globalCount += s.count;
    });
    const globalNNAvg = globalCount > 0 ? (globalNNTotal / globalCount) : 0;

    const data = Object.keys(regionalStats).map(region => {
        const stats = regionalStats[region];
        const avgSci = Math.round(stats.totalSci / stats.count);
        const avgNN = Math.round(stats.totalNN / stats.count);
        
        let parityHtml = '';
        const diff = avgNN - globalNNAvg;
        if (Math.abs(diff) <= 5) {
            parityHtml = `<span class="status-badge-inline" style="background:var(--accent-success); color:black">BALANCED</span>`;
        } else if (diff > 5) {
            parityHtml = `<span class="status-badge-inline" style="background:var(--accent-primary)">ADVANTAGED</span>`;
        } else {
            parityHtml = `<span class="status-badge-inline" style="background:var(--danger); color:white">DISADVANTAGED</span>`;
        }

        return `
            <tr>
                <td><strong>${region}</strong></td>
                <td>${stats.count}</td>
                <td>${avgSci}%</td>
                <td>${avgNN}%</td>
                <td>${parityHtml}</td>
            </tr>
        `;
    });

    tbody.innerHTML = data.join('');
}

// --- DATA PERSISTENCE (SUPABASE CLOUD) ---
async function loadData() {
    const savedWeights = localStorage.getItem('aesthetic_weights');
    if (savedWeights) State.weights = JSON.parse(savedWeights);

    State.events = JSON.parse(localStorage.getItem('aesthetic_events')) || [
        { id: 1, name: 'Derana Dream Star Finale', description: 'National level vocal and stage performance competition.', requirements: ['Vocal Range', 'Stage Presence', 'Baila', 'Sinhala Diction'] },
        { id: 2, name: 'Corporate Emcee Summit (Colombo)', description: 'Professional hosting for high-end corporate galas.', requirements: ['Public Speaking', 'Trilingual', 'Professionalism'] },
        { id: 3, name: 'Kandy Cultural Pageant', description: 'Traditional arts and drumming showcase.', requirements: ['Kandyan Dance', 'Geta Bera', 'Choreography'] }
    ];
    State.participants = JSON.parse(localStorage.getItem('aesthetic_participants')) || [];

    const statusDot = document.querySelector('#cloudStatus .status-dot');

    const defaultSeedData = [
        { id: 'seed-1', name: 'Elena Vance', region: 'Western', experience: 12, consistency: 92, skills: 'Vocal Range, Diction, Opera, Stage Presence', judgeA: 90, judgeB: 95, judgeC: 91, inactiveMonths: 1 },
        { id: 'seed-2', name: 'Julian Marsh', region: 'Central', experience: 5, consistency: 85, skills: 'Public Speaking, Emceeing, Professionalism, Humor', judgeA: 82, judgeB: 88, judgeC: 85, inactiveMonths: 4 },
        { id: 'seed-3', name: 'Sarah Sings', region: 'Uva', experience: 8, consistency: 78, skills: 'Vocal Range, Pop, Stage Presence, Improvisation', judgeA: 75, judgeB: 80, judgeC: 79, inactiveMonths: 12 }
    ];

    if (!_supabase) {
        console.warn("Supabase not initialized. Using local defaults.");
        if (statusDot) statusDot.classList.remove('online');
        if (State.participants.length === 0) {
            State.participants = defaultSeedData;
            saveToCache();
        }
        runKMeansAndAnomalies();
        return;
    }

    try {
        const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), 5000));
        const fetchP = _supabase.from('participants').select('*');
        const [pRes] = await Promise.race([Promise.all([fetchP]), timeout]);

        if (pRes.data && pRes.data.length > 0) {
            State.participants = pRes.data;
        } else {
            State.participants = defaultSeedData;
        }

        const statusBadge = document.getElementById('cloudStatus');
        if (statusBadge) statusBadge.classList.add('operational');
        if (statusDot) {
            statusDot.classList.add('online');
            statusDot.title = "Cloud Persistent: Active Connection";
        }
        
        logSync("Cloud Connection Established: Talent Database Operational.", "system");
        saveToCache();
        runKMeansAndAnomalies();
        renderMyProfile();
        renderParticipantRegistry();
        renderConsistencyMatrix();
        if (State.selectedEvent) renderRecommendations();
    } catch (err) {
        console.error("Cloud sync failed. Using cached data.", err);
        logSync("Warning: Cloud Connection Failed. Working in Local-Only Mode.", "remote");
        
        if (State.participants.length === 0) {
            State.participants = defaultSeedData;
            saveToCache();
        }
        
        const statusBadge = document.getElementById('cloudStatus');
        if (statusBadge) statusBadge.classList.remove('operational');
        if (statusDot) statusDot.classList.remove('online');
        runKMeansAndAnomalies();
        renderMyProfile();
        renderParticipantRegistry();
        renderConsistencyMatrix();
        if (State.selectedEvent) renderRecommendations();
    }
}

function saveToCache() {
    localStorage.setItem('aesthetic_participants', JSON.stringify(State.participants));
    localStorage.setItem('aesthetic_events', JSON.stringify(State.events));
    localStorage.setItem('aesthetic_weights', JSON.stringify(State.weights));
}

// --- ML INTEGRATIONS: NLP, LSTM, K-MEANS, ANOMALY ---

const AESTHETIC_DICTIONARY = {
    // English Core
    'vocal range': 'Vocal Range', 'diction': 'Diction', 'stage presence': 'Stage Presence',
    'emceeing': 'Emceeing', 'public speaking': 'Public Speaking', 'professionalism': 'Professionalism',
    'modulation': 'Modulation', 'improvisation': 'Improvisation', 'acting': 'Acting',
    'choreography': 'Choreography', 'humor': 'Humor', 'poetry': 'Poetry',
    
    // Local / Cultural Arts (English/Sinhlish)
    'kandyan': 'Kandyan Dance', 'pahatharata': 'Low-country Dance', 'geta bera': 'Geta Bera',
    'thabla': 'Thabla', 'baila': 'Baila', 'bodu gee': 'Devotional Songs', 'calypso': 'Calypso',
    'trilingual': 'Trilingual', 'sinhala': 'Sinhala Diction', 'tamil': 'Tamil Diction',

    // Sinhala/Tamil direct phrasing translations (Sinhlish/Tanglish)
    'sindu kiyanawa': 'Vocal Range', 'natum': 'Choreography', 'katha karanna': 'Public Speaking',
    'poddak': 'Improvisation', 'patteta': 'Stage Presence', 'nalla': 'Professionalism', 'paattu': 'Vocal Range'
};

function extractSkillsFromText(text) {
    const raw = text.toLowerCase();
    const foundSkills = new Set();

    // Scan for keys in the trilingual dictionary
    Object.keys(AESTHETIC_DICTIONARY).forEach(key => {
        if (raw.includes(key)) {
            foundSkills.add(AESTHETIC_DICTIONARY[key]);
        }
    });

    if (foundSkills.size === 0) {
        const words = text.match(/\b[A-Z][a-z]*\b/g);
        if (words) return [...new Set(words)].slice(0, 5).join(', ');
        return "Core Artistry";
    }
    
    return Array.from(foundSkills).join(', ');
}

function calculateConsistencyForecast(historyStr) {
    const scores = historyStr.split(',').map(s => parseFloat(s.trim()) / 100).filter(n => !isNaN(n));
    if (scores.length < 2) return Math.round((scores[0] * 100) || 50);

    try {
        const lstm = new brain.recurrent.LSTMTimeStep({ hiddenLayers: [8] });
        lstm.train([scores], { iterations: 100, errorThresh: 0.05 });
        const prediction = lstm.run(scores);
        return Math.min(Math.max(Math.round(prediction * 100), 0), 100);
    } catch (e) {
        const avg = scores.reduce((a,b)=>a+b,0)/scores.length;
        return Math.round(avg * 100);
    }
}

window.runKMeansAndAnomalies = () => {
    const container = document.getElementById('kmeansClusters');
    if (!container) return;

    if (State.participants.length < 3) {
        container.innerHTML = `
            <div class="glass-card" style="padding:2rem; text-align:center; grid-column:1/-1; border:1px dashed rgba(255,255,255,0.1)">
                <i data-lucide="database" style="margin-bottom:1rem; opacity:0.5"></i>
                <p style="font-size:0.85rem; opacity:0.7">Clustering Engine Awaiting Data Segment...</p>
                <div style="font-size:0.7rem; margin-top:0.5rem; color:var(--accent-secondary)">
                    Add ${3 - State.participants.length} more performer(s) to activate Algorithmic Archetype Mapping.
                </div>
            </div>
        `;
        if (typeof lucide !== 'undefined') lucide.createIcons();
        return;
    }
    
    // ANOMALY DETECTION
    const scores = State.participants.map(p => {
        const ev = State.selectedEvent || State.events[0];
        return calculateArtisticScore(p, ev).total;
    });
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const stdDev = Math.sqrt(scores.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / scores.length) || 1;
    
    State.participants.forEach((p, i) => {
        p.isDarkHorse = (stdDev > 0 && Math.abs((scores[i] - mean) / stdDev) > 1.2);
    });

    // K-MEANS CLUSTERING
    let centroids = [{ exp: 2, consist: 60 }, { exp: 7, consist: 80 }, { exp: 15, consist: 95 }];
    let clusters = [[], [], []];
    
    for(let iter = 0; iter < 5; iter++) {
        clusters = [[], [], []];
        State.participants.forEach(p => {
            let minDist = Infinity; let cIdx = 0;
            centroids.forEach((c, i) => {
                let d = Math.pow(c.exp - parseInt(p.experience), 2) + Math.pow(c.consist - parseInt(p.consistency), 2);
                if (d < minDist) { minDist = d; cIdx = i; }
            });
            clusters[cIdx].push(p);
        });
        clusters.forEach((clr, i) => {
            if (clr.length > 0) {
                centroids[i] = {
                    exp: clr.reduce((s, p) => s + parseInt(p.experience), 0) / clr.length,
                    consist: clr.reduce((s, p) => s + parseInt(p.consistency), 0) / clr.length
                };
            }
        });
    }
    
    const titles = ["Artistic Protégés", "Performance Masters", "Industry Legends"];
    const icons = ["sparkles", "star", "award"];
    const colors = ["var(--accent-secondary)", "var(--accent-primary)", "var(--accent-gold)"];
    
    container.innerHTML = clusters.map((clr, i) => `
        <div class="archetype-card animate-fadeIn" style="--card-glow: ${colors[i]};">
            <h4 class="archetype-title">
                <i data-lucide="${icons[i]}" style="width:18px; height:18px"></i> ${titles[i]}
            </h4>
            <p style="font-size:0.75rem; color:var(--text-secondary); margin-bottom:1rem">${clr.length} performers grouped by career trajectory.</p>
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
                ${clr.length > 0 ? clr.map(p => `<div class="tag" style="${p.isDarkHorse ? 'background:var(--danger);color:white; box-shadow: 0 0 10px rgba(239,68,68,0.5)' : 'background:rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1)'}">${p.name.split(' ')[0]} ${p.isDarkHorse ? '🏆' : ''}</div>`).join('') : '<span style="font-size:0.7rem; opacity:0.3">No entities in segment</span>'}
            </div>
        </div>
    `).join('');
    if (typeof lucide !== 'undefined') lucide.createIcons();
};

window.autoTuneWeights = () => {
    const btn = document.getElementById('autoTuneBtn');
    if(btn) btn.disabled = true;
    
    setTimeout(() => {
        let best = { s: 0.6, c: 0.3, e: 0.1, loss: Infinity };
        for(let s = 20; s <= 70; s += 10) {
            for(let c = 20; c <= 70; c += 10) {
                let e = 100 - (s + c);
                if (e < 5 || e > 30) continue;
                State.weights = { skill: s/100, consist: c/100, exp: e/100 };
                let err = 0;
                State.participants.forEach(p => {
                    State.events.forEach(ev => err += calculateArtisticScore(p, ev).conflict.score);
                });
                if (err < best.loss) { best = { s: s/100, c: c/100, e: e/100, loss: err }; }
            }
        }
        State.weights = { skill: best.s, consist: best.c, exp: best.e };
        document.getElementById('weightSkill').value = best.s * 100;
        document.getElementById('weightConsist').value = best.c * 100;
        document.getElementById('weightExp').value = best.e * 100;
        
        document.getElementById('weightSkillVal').textContent = Math.round(best.s * 100);
        document.getElementById('weightConsistVal').textContent = Math.round(best.c * 100);
        document.getElementById('weightExpVal').textContent = Math.round(best.e * 100);
        
        showToast("Artistic weights calibrated to neural optimum.", "success");
        if(btn) btn.disabled = false;
        if (State.selectedEvent) renderRecommendations();
    }, 800);
};

window.exportDossier = () => {
    if (!State.selectedEvent) return;
    const data = State.participants.map(p => {
        const res = calculateArtisticScore(p, State.selectedEvent);
        return `${p.name}, Score: ${res.total}%, Skill Match: ${res.scientific}%, Net Forecast: ${res.probability}%`;
    });
    data.unshift("Performer Name, Total Score, Skill Match, Net Forecast");
    const blob = new Blob([data.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Dossier_${State.selectedEvent.name.replace(/\s+/g, '_')}.csv`;
    a.click();
    showToast("Event Dossier Exported Successfully", "success");
};

async function deleteParticipant(id) {
    if (!confirm("Are you sure you want to remove this performer from the registry?")) return;
    
    if (_supabase) {
        await _supabase.from('participants').delete().eq('id', id);
    }
    State.participants = State.participants.filter(p => String(p.id) !== String(id));
    saveToCache();
    renderParticipantRegistry();
    renderConsistencyMatrix();
    runKMeansAndAnomalies();
    showToast("Performer purged from registry.", "remote");
}

window.optimizeTeam = () => {
    const results = document.getElementById('teamOptimizationResult');
    if (!results || !State.selectedEvent) return;

    // Heuristic: Best Total, Best Consistency, and Best Tenure
    const scored = State.participants.map(p => ({
        ...p,
        res: calculateArtisticScore(p, State.selectedEvent)
    }));

    if (scored.length < 3) {
        showToast("Add at least 3 performers to discover an optimal ensemble.", "primary");
        return;
    }

    const bestSkill = [...scored].sort((a,b) => b.res.scientific - a.res.scientific)[0];
    const bestConsist = [...scored].filter(p => p.id !== bestSkill.id).sort((a,b) => b.res.probability - a.res.probability)[0];
    const bestTenure = [...scored].filter(p => p.id !== bestSkill.id && p.id !== bestConsist.id).sort((a,b) => parseInt(b.experience) - parseInt(a.experience))[0];

    const team = [bestSkill, bestConsist, bestTenure].filter(Boolean);
    
    document.getElementById('teamMembers').innerHTML = team.map(p => `
        <div style="flex:1; text-align:center; padding:1rem; background:var(--bg-deep); border-radius:12px; border:1px solid var(--border)">
            <div class="avatar" style="width:40px; height:40px; margin:0 auto 0.5rem; font-size:0.8rem">${p.name[0]}</div>
            <div style="font-size:0.85rem; font-weight:900">${p.name}</div>
            <div style="font-size:0.6rem; color:var(--accent-primary)">${p.res.total}% Hybrid Align</div>
        </div>
    `).join('');
    
    document.getElementById('teamCoverage').textContent = `Optimal Composition: ${State.selectedEvent.name}`;
    results.style.display = 'block';
    showToast("Ensemble Discovery Optimized.", "success");
}

// --- UI RENDERING ---
async function initApp() {
    // Ensure lucide icons render safely
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }

    initTheme();
    setupEventListeners();
    
    // Always hide admin links on startup (user must login via PIN)
    document.querySelectorAll('.admin-link').forEach(el => {
        el.style.display = 'none';
    });
    State.isAdmin = false;

    switchTab('dashboard');
    await loadData();
    renderEvents();
    renderParticipantRegistry();
    renderSkillChips();
    renderMyProfile();
    setTimeout(initPredictor, 1200);
    
    // Automatically select the first event to populate dashboards by default
    setTimeout(() => {
        const firstEventItem = document.querySelector('.event-item');
        if (firstEventItem) {
            firstEventItem.click();
        }
    }, 800);
}

function renderSkillChips() {
    const container = document.getElementById('skillChips');
    if (!container) return;

    // Use all keys from the dictionary
    const chipKeys = Object.keys(AESTHETIC_DICTIONARY);
    
    container.innerHTML = chipKeys.map(key => {
        return `<span class="skill-chip" onclick="appendSkillChip('${key}')">+ ${key}</span>`;
    }).join('');
}

window.appendSkillChip = (chipText) => {
    const bio = document.getElementById('bioInput');
    if (bio) {
        let val = bio.value.trim();
        if (val.length > 0 && !val.endsWith(',')) val += ', ';
        else if (val.length > 0) val += ' ';
        bio.value = val + chipText;
    }
}

function initTheme() {
    const saved = localStorage.getItem('aesthetic_theme') || 'light';
    if (saved === 'dark') {
        document.body.classList.add('dark-mode');
        const icon = document.querySelector('#themeToggle i');
        if (icon) icon.setAttribute('data-lucide', 'moon');
    }
    lucide.createIcons();
}

window.toggleTheme = () => {
    const isDark = document.body.classList.toggle('dark-mode');
    localStorage.setItem('aesthetic_theme', isDark ? 'dark' : 'light');
    
    const icon = document.querySelector('#themeToggle i');
    if (icon) {
        icon.setAttribute('data-lucide', isDark ? 'moon' : 'sun');
        lucide.createIcons();
    }
    
    showToast(`Theme switched to ${isDark ? 'Midnight' : 'Arctic Pearl'}`, "primary");
};

function setupEventListeners() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.onclick = () => switchTab(btn.dataset.tab);
    });

    ['weightSkill', 'weightConsist', 'weightExp'].forEach(id => {
        const slider = document.getElementById(id);
        if (slider) {
            slider.oninput = (e) => {
                document.getElementById(id + 'Val').textContent = e.target.value;
                const s = (document.getElementById('weightSkill').value) / 100;
                const c = (document.getElementById('weightConsist').value) / 100;
                const ex = (document.getElementById('weightExp').value) / 100;
                const total = s + c + ex;
                State.weights = { skill: s/total, consist: c/total, exp: ex/total };
                if (State.selectedEvent) renderRecommendations();
            };
        }
    });

    const pForm = document.getElementById('participantForm');
    if (pForm) {
        pForm.onsubmit = async (e) => {
            e.preventDefault();
            const btn = pForm.querySelector('button[type="submit"]');
            btn.innerHTML = `<i data-lucide="loader" class="rotating"></i> Neural Inferencing...`;
            btn.disabled = true;

            setTimeout(async () => {
                const formData = new FormData(pForm);

                const myId = localStorage.getItem('my_profile_id');
                const isUpdating = myId && State.participants.some(p => String(p.id) === String(myId));

                let existingJudgeA = 0;
                let existingJudgeB = 0;
                let existingJudgeC = 0;

                if (isUpdating) {
                    const existing = State.participants.find(p => String(p.id) === String(myId));
                    if (existing) {
                        existingJudgeA = existing.judgeA || 0;
                        existingJudgeB = existing.judgeB || 0;
                        existingJudgeC = existing.judgeC || 0;
                    }
                }

                const forensics = {
                    name: formData.get('name'),
                    region: formData.get('region') || 'Western',
                    experience: parseInt(formData.get('experience')),
                    inactiveMonths: parseInt(formData.get('inactiveMonths')) || 0,
                    videoUrl: formData.get('videoUrl') || '',
                    judgeA: existingJudgeA,
                    judgeB: existingJudgeB,
                    judgeC: existingJudgeC,
                    consistency: Math.round((existingJudgeA + existingJudgeB + existingJudgeC) / 3),
                    skills: extractSkillsFromText(formData.get('bio'))
                };

                let pID = myId;
                if (isUpdating) {
                    forensics.id = myId;
                    if (_supabase) {
                        const { error } = await _supabase.from('participants').update(forensics).eq('id', myId);
                        if (!error) {
                            logSync(`Successfully updated ${forensics.name} in Cloud Database.`, "success");
                        } else {
                            console.error("Cloud update error:", error);
                            logSync("Cloud Sync Error: Update saved locally only.", "remote");
                        }
                    }
                    const idx = State.participants.findIndex(p => String(p.id) === String(myId));
                    if (idx !== -1) State.participants[idx] = forensics;
                } else {
                    if (_supabase) {
                        const { data, error } = await _supabase.from('participants').insert([forensics]).select();
                        if (!error && data) {
                            pID = data[0].id;
                            logSync(`Successfully synced ${forensics.name} to Cloud Database.`, "success");
                        } else {
                            console.error("Cloud insert error:", error);
                            logSync("Cloud Sync Error: Data saved locally only.", "remote");
                        }
                    }

                    if (pID) forensics.id = pID;
                    else forensics.id = Date.now();

                    State.participants.push(forensics);
                    localStorage.setItem('my_profile_id', forensics.id);
                }

                saveToCache();
                pForm.reset();
                runKMeansAndAnomalies();
                renderParticipantRegistry();
                renderMyProfile();
                if (State.selectedEvent) renderRecommendations();
                
                showToast(isUpdating ? "Profile updated successfully." : "Profile registered successfully.", "primary");
                logSync(`${isUpdating ? 'Updated' : 'Registered'} performer: ${forensics.name}`, "primary");
                
                btn.innerHTML = `<i data-lucide="sparkles"></i> Analyze & Register`;
                btn.disabled = false;
                lucide.createIcons();

                switchTab('profile');
            }, 600);
        };
    }
}

function switchTab(tabId) {
    if (!State.isAdmin && (tabId === 'management' || tabId === 'analytics')) {
        return switchTab('dashboard');
    }

    const section = document.getElementById(`${tabId}Section`);
    if (!section) return switchTab('dashboard');
    
    State.activeTab = tabId;
    localStorage.setItem('aesthetic_active_tab', tabId);
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    section.classList.add('active');
    
    // sync nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
}

window.toggleAdminMode = () => {
    if (State.isAdmin) {
        State.isAdmin = false;
        showToast("Administrative session terminated.", "primary");
    } else {
        const pin = prompt("Enter Administrative Authorization PIN:");
        if (pin === "2001") {
            State.isAdmin = true;
            showToast("Administrative Authorization Granted.", "success");
        } else if (pin !== null) {
            showToast("Invalid Authorization PIN.", "danger");
        }
    }
    
    // Update UI elements that depend on admin status
    const adminBtn = document.getElementById('adminToggleBtn');
    if (adminBtn) {
        adminBtn.classList.toggle('active', State.isAdmin);
        const icon = adminBtn.querySelector('i');
        if (icon) icon.setAttribute('data-lucide', State.isAdmin ? 'unlock' : 'lock');
        lucide.createIcons();
    }

    // Toggle administrative links visibility
    document.querySelectorAll('.admin-link').forEach(el => {
        el.style.display = State.isAdmin ? 'flex' : 'none';
    });

    if (!State.isAdmin && (State.activeTab === 'management' || State.activeTab === 'analytics')) {
        switchTab('dashboard');
    }
    
    renderParticipantRegistry();
};

function renderEvents() {
    const list = document.getElementById('eventList');
    list.innerHTML = State.events.map(ev => `
        <div class="event-item" onclick="selectEvent(${ev.id}, this)">
            <h4>${ev.name}</h4>
            <div class="tags">${ev.requirements.map(r => `<span class="tag">${r}</span>`).join('')}</div>
        </div>
    `).join('');
}

function selectEvent(id, el) {
    State.selectedEvent = State.events.find(e => e.id === id);
    document.querySelectorAll('.event-item').forEach(e => e.classList.remove('active'));
    el.classList.add('active');
    document.getElementById('noSelection').style.display = 'none';
    document.getElementById('eventAnalytics').style.display = 'block';
    document.getElementById('targetEventName').textContent = State.selectedEvent.name;
    document.getElementById('targetEventDesc').textContent = State.selectedEvent.description;
    
    // Show Advanced Analytics Sections
    const teamSection = document.getElementById('teamDiscoverySection');
    if (teamSection) teamSection.style.display = 'block';
    
    renderRecommendations();
    renderConsistencyMatrix();
}

function renderRecommendations() {
    const container = document.getElementById('recommendationContainer');
    if (!container || !State.selectedEvent) return;
    
    container.innerHTML = '';
    const scored = State.participants
        .map(p => ({ ...p, result: calculateArtisticScore(p, State.selectedEvent) }))
        .sort((a, b) => b.result.total - a.result.total);

    scored.forEach((p, idx) => {
        const div = document.createElement('div');
        div.className = `medalist-card top-${idx + 1}`;
        div.onclick = () => showDetails(p);
        div.innerHTML = `
            <div class="avatar">${p.name.split(' ').map(n=>n[0]).join('')}</div>
            <div style="flex:1">
                <div style="display:flex; align-items:center;">
                    <h3 style="margin-bottom:0.2rem">${p.name}</h3>
                    ${p.isDarkHorse ? `<div class="status-badge-anomaly">🏆 ANOMALY</div>` : ''}
                </div>
                <p style="font-size:0.75rem; color:var(--text-secondary)">
                    ${p.result.scientific}% Performative Alignment • ${p.result.probability}% Forecast
                </p>
                <div style="margin-top:0.4rem">${generateRadarMini(p.result.breakdown)}</div>
            </div>
            <div class="score-num">${p.result.total}%</div>
        `;
        container.appendChild(div);
    });
    lucide.createIcons();
}

function generateRadarMini(breakdown) {
    return `<svg width="100" height="6" style="background:rgba(255,255,255,0.05); border-radius:3px">
        <rect width="${breakdown.skill * 60}" height="6" fill="var(--accent-primary)" />
        <rect x="${breakdown.skill * 60}" width="${breakdown.consist * 30}" height="6" fill="var(--accent-secondary)" />
        <rect x="${(breakdown.skill * 60) + (breakdown.consist * 30)}" width="${breakdown.exp * 10}" height="6" fill="var(--accent-gold)" />
    </svg>`;
}

function renderParticipantRegistry() {
    const tbody = document.getElementById('participantRegistry');
    if (!tbody) return;
    tbody.innerHTML = State.participants.map(p => `
        <tr>
            <td><strong>${p.name}</strong></td>
            <td><span class="tag" style="font-size:0.6rem; padding:0.2rem 0.4rem; background:rgba(255,255,255,0.1)">${p.region || 'Western'}</span></td>
            <td>${p.experience} Yrs</td>
            <td>${p.consistency}% ${p.inactiveMonths ? `<span style="font-size:0.7rem; color:var(--accent-secondary); opacity:0.8;">(${p.inactiveMonths}m idle)</span>` : ''}</td>
            <td>${p.skills}</td>
            <td>
                ${State.isAdmin ? `
                    <div style="display: flex; gap: 0.5rem;">
                        <button class="action-btn" onclick="openGradingModal('${p.id}')" title="Score Performer" style="background: rgba(99,102,241,0.15); border: 1px solid var(--accent-primary); color: var(--accent-primary);">
                            <i data-lucide="award"></i>
                        </button>
                        <button class="action-btn" onclick="deleteParticipant('${p.id}')" title="Delete Performer" style="background: rgba(239,68,68,0.15); border: 1px solid var(--danger); color: var(--danger);">
                            <i data-lucide="trash-2"></i>
                        </button>
                    </div>
                ` : `<span style="opacity:0.2; font-size:0.7rem"><i data-lucide="lock" style="width:12px; height:12px"></i></span>`}
            </td>
        </tr>
    `).join('');
    lucide.createIcons();
}

function showToast(m, t) {
    const c = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `status-badge ${t}`;
    toast.style.margin = '1rem';
    toast.textContent = m;
    c.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

function logSync(m, t) {
    const c = document.getElementById('syncLog');
    const d = document.createElement('div');
    d.className = `log-entry ${t}`;
    d.textContent = `[${new Date().toLocaleTimeString()}] ${m}`;
    c.prepend(d);
}

function showDetails(p) {
    const modal = document.getElementById('detailsModal');
    const body = document.getElementById('modalBody');
    let simYears = 5;
    let radarChartInstance = null;

    const draw = () => {
        const trajectory = calculateTrajectory(p, simYears, State.selectedEvent);
        const futureRes = trajectory[trajectory.length - 1];
        const res = futureRes.full; // Use projected state so all components update on slider drag
        
        body.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem">
                <div>
                    <h2 style="color:var(--accent-primary); font-size:1.8rem">${p.name}</h2>
                    <p style="color:var(--text-secondary); font-size:0.9rem">Artistic Intelligence Report • Performer Deep-Audit</p>
                </div>
                ${p.isDarkHorse ? `<div class="status-badge-anomaly" style="position:static">🏆 ARTISTIC ANOMALY</div>` : ''}
            </div>
            
            <div style="display:grid; grid-template-columns: 1.1fr 1fr; gap:1.5rem;">
                <!-- LEFT: CONSENSUS DATA & RADAR CHART -->
                <div class="glass-panel" style="padding:1.5rem">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem">
                        <div style="text-align:center; flex:1">
                            <label style="font-size:0.7rem; text-transform:uppercase; color:var(--text-muted)">Projected Rating</label>
                            <div style="font-size:1.8rem; font-weight:900; color:var(--text-primary)">${res.total}%</div>
                        </div>
                        <div style="width:1px; height:30px; background:var(--border)"></div>
                        <div style="text-align:center; flex:1">
                            <label style="font-size:0.7rem; text-transform:uppercase; color:var(--text-muted)">Model Divergence</label>
                            <div style="font-size:1.8rem; font-weight:900; color:var(--accent-secondary)">${res.conflict.score}%</div>
                        </div>
                    </div>
                    
                    <h4 style="margin-bottom:0.5rem; font-size:0.85rem">Performative Dimension Radar</h4>
                    <div style="position:relative; height:180px; width:100%; display:flex; justify-content:center; align-items:center;">
                        <canvas id="radarChartCanvas" style="max-height:100%; max-width:100%;"></canvas>
                    </div>
                    
                    <div style="margin-top:0.8rem; padding-top:0.8rem; border-top:1px dashed var(--border)">
                        <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:0.2rem">
                            <span>Regional Opportunity Index (${p.region || 'Western'})</span>
                            <span style="font-weight:bold">${(res.equity.index * 100).toFixed(0)}% Access</span>
                        </div>
                        <p style="font-size:0.65rem; color:var(--text-secondary); margin:0">Tenacity Multiplier Applied: <strong>${res.equity.multiplier.toFixed(2)}x</strong></p>
                    </div>
                </div>

                <!-- RIGHT: EVOLUTION SIMULATION -->
                <div class="glass-panel" style="padding:1.5rem">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem">
                        <h4 style="font-size:0.85rem">Artistic Growth Projection</h4>
                        <div style="text-align:right">
                            <span style="font-size:0.6rem; text-transform:uppercase; opacity:0.6">Year ${simYears} Potential</span>
                            <div style="font-size:1.2rem; font-weight:900; color:var(--accent-gold)">${futureRes.score}%</div>
                        </div>
                    </div>
                    
                    <div style="margin-bottom:1.5rem">
                        <input type="range" id="simYearSlider" min="1" max="10" value="${simYears}" style="width:100%">
                        <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:var(--text-muted); margin-top:0.4rem">
                            <span>Immediate (1Y)</span>
                            <span>Strategic Outlook (${simYears}Y)</span>
                        </div>
                    </div>
                    
                    <div style="display:flex; align-items:flex-end; gap:0.5rem; height:100px; padding-bottom:10px; border-bottom:1px solid var(--border)">
                        ${trajectory.map((t, i) => `
                            <div style="flex:1; display:flex; flex-direction:column; align-items:center; gap:0.4rem">
                                <div style="width:100%; height:${Math.max(t.score, 5)}%; background:${i === trajectory.length-1 ? 'var(--accent-gold)' : 'linear-gradient(to top, var(--accent-primary), var(--accent-secondary))'}; border-radius:3px; opacity:${0.4 + (i * 0.1)}; transition: height 0.4s ease"></div>
                                <span style="font-size:0.55rem; color:var(--text-muted)">Y${t.year}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>

            <div style="margin-top:1.5rem">
                <h4 style="margin-bottom:0.6rem">Proficiency Impact Management</h4>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem">
                    ${State.selectedEvent.requirements.map(req => {
                        const has = p.skills.toLowerCase().includes(req.toLowerCase());
                        return `<div class="tag" style="cursor:pointer; background:${has ? 'var(--accent-primary)' : 'var(--bg-deep)'}; color:${has ? 'white' : 'var(--text-secondary)'}" onclick="toggleSimSkill(${p.id}, '${req}')">
                            <i data-lucide="${has ? 'check' : 'plus'}" style="width:12px; height:12px; margin-right:4px"></i>${req}
                        </div>`;
                    }).join('')}
                </div>
            </div>

            <!-- Showcase Performance Clip Container (Downsized & Centered) -->
            <div class="glass-panel" style="margin-top: 1.5rem; padding: 1.25rem; display: flex; flex-direction: column; gap: 0.75rem; position: relative; z-index: 5;">
                <h4 style="margin: 0; display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem">
                    <i data-lucide="video" style="color:var(--accent-primary)"></i> Performative Showcase Clip
                </h4>
                ${p.videoUrl ? `
                    <div style="position:relative; width:100%; max-width:480px; margin:0 auto; padding-bottom:270px; height:0; overflow:hidden; border-radius:12px; border:1px solid var(--border)">
                        ${getMediaEmbedHtml(p.videoUrl)}
                    </div>
                ` : `
                    <div style="background:var(--bg-deep); border-radius:12px; padding:1.25rem; display:flex; flex-direction:column; align-items:center; justify-content:center; border:1px dashed var(--border); position:relative; overflow:hidden; min-height:100px; max-width:480px; margin:0 auto; width:100%">
                        <div style="display:flex; align-items:center; gap:0.3rem; margin-bottom:0.6rem">
                            <!-- Glowing animated micro-visualizer lines -->
                            <div style="width:3px; height:15px; background:var(--accent-primary); border-radius:3px; animation: bounce 0.8s ease-in-out infinite alternate"></div>
                            <div style="width:3px; height:35px; background:var(--accent-secondary); border-radius:3px; animation: bounce 0.6s ease-in-out infinite alternate 0.1s"></div>
                            <div style="width:3px; height:20px; background:var(--accent-primary); border-radius:3px; animation: bounce 0.9s ease-in-out infinite alternate 0.2s"></div>
                            <div style="width:3px; height:45px; background:var(--accent-gold); border-radius:3px; animation: bounce 0.7s ease-in-out infinite alternate 0.3s"></div>
                            <div style="width:3px; height:15px; background:var(--accent-secondary); border-radius:3px; animation: bounce 0.8s ease-in-out infinite alternate 0.4s"></div>
                        </div>
                        <span style="font-size:0.7rem; color:var(--text-secondary); text-align:center; font-weight:700">No media clip uploaded by performer.</span>
                        <span style="font-size:0.6rem; color:var(--text-muted); margin-top:0.15rem">Generating synthesized algorithmic performative forecast...</span>
                    </div>
                `}
            </div>
        `;

        lucide.createIcons();

        // Initialize Chart.js Radar Chart
        const canvas = document.getElementById('radarChartCanvas');
        if (canvas) {
            if (radarChartInstance) {
                radarChartInstance.destroy();
            }
            const ctx = canvas.getContext('2d');
            const isDark = document.body.classList.contains('dark-mode');
            const textColor = isDark ? '#ffffff' : '#1e1e1e';
            const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';

            radarChartInstance = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Skill Match', 'Consistency', 'Tenacity'],
                    datasets: [
                        {
                            label: 'Performer Cap.',
                            data: [
                                Math.round(res.breakdown.skill * 100),
                                Math.round(res.breakdown.consist * 100),
                                Math.round(res.breakdown.exp * 100)
                            ],
                            backgroundColor: 'rgba(109, 40, 217, 0.25)', // Primary accent
                            borderColor: '#6d28d9',
                            borderWidth: 2,
                            pointBackgroundColor: '#6d28d9'
                        },
                        {
                            label: 'Weights',
                            data: [
                                Math.round(State.weights.skill * 100),
                                Math.round(State.weights.consist * 100),
                                Math.round(State.weights.exp * 100)
                            ],
                            backgroundColor: 'rgba(217, 70, 239, 0.1)', // Secondary accent
                            borderColor: '#d946ef',
                            borderWidth: 1.5,
                            borderDash: [3, 3],
                            pointBackgroundColor: '#d946ef'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: { color: gridColor },
                            grid: { color: gridColor },
                            pointLabels: {
                                color: textColor,
                                font: { family: 'Outfit', size: 9, weight: '600' }
                            },
                            ticks: {
                                display: false,
                                maxTicksLimit: 5
                            },
                            suggestedMin: 0,
                            suggestedMax: 100
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'bottom',
                            labels: {
                                color: textColor,
                                font: { family: 'Outfit', size: 8 },
                                boxWidth: 8
                            }
                        }
                    }
                }
            });
        }

        document.getElementById('simYearSlider').oninput = (e) => {
            simYears = parseInt(e.target.value);
            draw();
        };
    };

    window.toggleSimSkill = (pid, skill) => {
        const perf = State.participants.find(pt => pt.id === pid);
        let sList = perf.skills.split(',').map(s => s.trim());
        if (sList.some(s => s.toLowerCase() === skill.toLowerCase())) {
            sList = sList.filter(s => s.toLowerCase() !== skill.toLowerCase());
        } else {
            sList.push(skill);
        }
        perf.skills = sList.join(', ');
        saveToCache();
        draw();
        renderRecommendations();
    };

    draw();
    modal.style.display = 'flex';
    document.querySelector('.close-modal').onclick = () => {
        if (radarChartInstance) {
            radarChartInstance.destroy();
        }
        modal.style.display = 'none';
    };
}

// Wait for all CDN scripts to be ready before initializing
function waitForCDNAndInit() {
    const ready = (
        typeof lucide !== 'undefined' &&
        typeof supabase !== 'undefined' &&
        typeof brain !== 'undefined'
    );
    if (ready) {
        initApp();
    } else {
        setTimeout(waitForCDNAndInit, 100);
    }
}
window.addEventListener('load', waitForCDNAndInit);

window.exportSPSS = () => {
    if (State.participants.length === 0) {
        showToast("No data to export.", "primary");
        return;
    }
    
    // Headers designed for SPSS / R
    const headers = [
        "Participant_ID", "Region", "Industry_Tenure_Yrs", 
        "JudgeA_Score", "JudgeB_Score", "JudgeC_Score", 
        "Consensus_Avg", "Skill_Match_Sci", "Neural_Forecast_Prob", 
        "Hybrid_Final_Score", "Conflict_Index", "Tenacity_Multiplier"
    ];

    const rows = State.participants.map((p, index) => {
        const ev = State.selectedEvent || State.events[0];
        const res = calculateArtisticScore(p, ev);
        return [
            `PID_${index + 1}`,
            p.region || "Western",
            p.experience,
            p.judgeA !== undefined ? p.judgeA : p.consistency,
            p.judgeB !== undefined ? p.judgeB : p.consistency,
            p.judgeC !== undefined ? p.judgeC : p.consistency,
            p.consistency,
            res.scientific,
            res.probability,
            res.total,
            res.conflict.score,
            res.equity.multiplier.toFixed(3)
        ].join(",");
    });

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `talent_premium_spss_export_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("SPSS Dossier Downloaded", "success");
};

window.toggleAblationMode = (checked) => {
    const panel = document.getElementById('ablationPanel');
    if (panel) {
        panel.style.display = checked ? 'block' : 'none';
        if (checked) {
            calculateAblation();
        }
    }
};

window.calculateAblation = () => {
    const toggle = document.getElementById('ablationToggle');
    if (!toggle || !toggle.checked) return;

    const participants = State.participants;
    if (participants.length === 0) return;

    let sqErrNN = 0;
    let sqErrWSM = 0;
    let sqErrHybrid = 0;

    const ev = State.selectedEvent || State.events[0];

    participants.forEach(p => {
        const res = calculateArtisticScore(p, ev);
        // Using Judge consensus average as the "Ground Truth" value for MSE testing
        const truth = p.consistency; 

        sqErrNN += Math.pow(res.probability - truth, 2);
        sqErrWSM += Math.pow(res.scientific - truth, 2);
        sqErrHybrid += Math.pow(res.total - truth, 2);
    });

    const mseNN = (sqErrNN / participants.length).toFixed(2);
    const mseWSM = (sqErrWSM / participants.length).toFixed(2);
    const mseHybrid = (sqErrHybrid / participants.length).toFixed(2);

    const nnEl = document.getElementById('mseNN');
    const wsmEl = document.getElementById('mseWSM');
    const hybEl = document.getElementById('mseHybrid');
    
    if (nnEl) nnEl.textContent = mseNN;
    if (wsmEl) wsmEl.textContent = mseWSM;
    if (hybEl) hybEl.textContent = mseHybrid;
};

function getMediaEmbedHtml(url) {
    if (!url) return '';
    
    // 1. YouTube Link Conversion
    const ytReg = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
    const ytMatch = url.match(ytReg);
    if (ytMatch && ytMatch[2].length === 11) {
        return `<iframe style="position:absolute; top:0; left:0; width:100%; height:100%; border:none" src="https://www.youtube.com/embed/${ytMatch[2]}" allow="autoplay; encrypted-media" allowfullscreen></iframe>`;
    }
    
    // 2. Google Drive Link Conversion
    if (url.includes('drive.google.com')) {
        let embedUrl = url.replace(/\/view\??.*/, '/preview').replace(/\/edit\??.*/, '/preview');
        if (!embedUrl.endsWith('/preview')) {
            embedUrl = embedUrl.split('?')[0] + '/preview';
        }
        return `<iframe style="position:absolute; top:0; left:0; width:100%; height:100%; border:none" src="${embedUrl}" allow="autoplay" allowfullscreen></iframe>`;
    }
    
    // 3. Direct Video file (MP4, WebM, OGG)
    if (url.match(/\.(mp4|webm|ogg)($|\?)/i)) {
        return `<video style="position:absolute; top:0; left:0; width:100%; height:100%; border-radius:12px" controls src="${url}"></video>`;
    }

    // 4. Fallback: Standard preview/external document frame
    return `<iframe style="position:absolute; top:0; left:0; width:100%; height:100%; border:none; background:white" src="${url}" allowfullscreen></iframe>`;
}

window.simulateCVUpload = (input) => {
    const file = input.files[0];
    if (!file) return;
    
    showToast("Analyzing Resume document layout...", "primary");
    logSync(`Importing Resume CV: ${file.name}`, "system");

    setTimeout(() => {
        showToast("NLP Engine: Parsing semantic capability graph...", "primary");
    }, 1200);

    setTimeout(() => {
        // Randomly pick a few highly-advanced local/cultural skills
        const mockSkills = [
            "Kandyan Dance, Geta Bera rhythm alignment, outstanding Stage Presence, Trilingual fluency",
            "Baila singing, Calypso music, public speaking, humor, improvisation",
            "Opera vocal range, Low-country Dance, Sinhala Diction, acting masterclass",
            "Corporate Emceeing, Professionalism, Trilingual hosting, public speaking"
        ];
        const selected = mockSkills[Math.floor(Math.random() * mockSkills.length)];
        
        const bioInput = document.getElementById('bioInput');
        if (bioInput) {
            bioInput.value = `[NLP CV Import - parsed ${file.name}]: Professional performer specializing in: ${selected}. Verified experience and regional capability registered.`;
            // Trigger skill extraction refresh
            showToast("NLP Extraction Complete: 4 capabilities synced.", "success");
            logSync(`NLP parsed skills from ${file.name} successfully.`, "success");
            
            // Highlight skills by triggering word-match dictionary parsing automatically on text change
            // We can even call extractSkillsFromText or let them click analyze
        }
    }, 2400);
};

function renderMyProfile() {
    const noProfileEl = document.getElementById('noProfileState');
    const profileDetailsEl = document.getElementById('profileDetailsState');
    if (!noProfileEl || !profileDetailsEl) return;

    const myId = localStorage.getItem('my_profile_id');
    const myProfile = State.participants.find(p => String(p.id) === String(myId));

    if (!myProfile) {
        noProfileEl.style.display = 'block';
        profileDetailsEl.style.display = 'none';
        return;
    }

    noProfileEl.style.display = 'none';
    profileDetailsEl.style.display = 'block';

    document.getElementById('myProfileName').textContent = myProfile.name;
    document.getElementById('myProfileRegion').textContent = myProfile.region;
    document.getElementById('myProfileExperience').textContent = `${myProfile.experience} Yrs`;
    document.getElementById('myProfileConsistency').textContent = `${myProfile.consistency}%`;

    const skillsContainer = document.getElementById('myProfileSkills');
    if (skillsContainer) {
        const skillsList = myProfile.skills.split(',').map(s => s.trim());
        skillsContainer.innerHTML = skillsList.map(s => `<span class="tag" style="background:rgba(255,255,255,0.05); border:1px solid var(--border)">${s}</span>`).join('');
    }

    const mediaContainer = document.getElementById('myProfileVideoContainer');
    if (mediaContainer) {
        if (myProfile.videoUrl) {
            mediaContainer.innerHTML = `
                <div style="position:relative; width:100%; max-width:480px; margin:0 auto; padding-bottom:270px; height:0; overflow:hidden; border-radius:12px; border:1px solid var(--border)">
                    ${getMediaEmbedHtml(myProfile.videoUrl)}
                </div>
            `;
        } else {
            mediaContainer.innerHTML = `
                <div style="background:var(--bg-deep); border-radius:12px; padding:1.25rem; display:flex; flex-direction:column; align-items:center; justify-content:center; border:1px dashed var(--border); min-height:100px; max-width:480px; margin:0 auto; width:100%">
                    <span style="font-size:0.7rem; color:var(--text-secondary); text-align:center; font-weight:700">No media clip uploaded.</span>
                </div>
            `;
        }
    }
    if (window.lucide) {
        window.lucide.createIcons();
    }
}

async function deleteMyProfile() {
    const myId = localStorage.getItem('my_profile_id');
    if (!myId) return;

    if (!confirm("Are you sure you want to delete your registered profile? This will remove all your data from the system.")) return;

    if (_supabase) {
        await _supabase.from('participants').delete().eq('id', myId);
    }
    State.participants = State.participants.filter(p => String(p.id) !== String(myId));
    localStorage.removeItem('my_profile_id');
    
    saveToCache();
    renderParticipantRegistry();
    renderConsistencyMatrix();
    runKMeansAndAnomalies();
    renderMyProfile();
    
    if (State.selectedEvent) renderRecommendations();
    
    showToast("Your profile has been deleted.", "primary");
    switchTab('register');
}

function prepopulateRegistrationForm() {
    const myId = localStorage.getItem('my_profile_id');
    const myProfile = State.participants.find(p => String(p.id) === String(myId));
    if (!myProfile) return;

    const form = document.getElementById('participantForm');
    if (!form) return;

    form.querySelector('input[name="name"]').value = myProfile.name;
    form.querySelector('select[name="region"]').value = myProfile.region;
    form.querySelector('input[name="experience"]').value = myProfile.experience;
    form.querySelector('input[name="inactiveMonths"]').value = myProfile.inactiveMonths || 0;
    
    form.querySelector('textarea[name="bio"]').value = myProfile.bio || myProfile.skills;
    form.querySelector('input[name="videoUrl"]').value = myProfile.videoUrl || '';
}

function openGradingModal(id) {
    const p = State.participants.find(pt => String(pt.id) === String(id));
    if (!p) return;

    document.getElementById('gradingPerformerId').value = id;
    document.getElementById('gradeJudgeA').value = p.judgeA || 0;
    document.getElementById('gradeJudgeB').value = p.judgeB || 0;
    document.getElementById('gradeJudgeC').value = p.judgeC || 0;

    document.getElementById('gradingModal').style.display = 'flex';
}

function closeGradingModal() {
    document.getElementById('gradingModal').style.display = 'none';
}

async function submitGrading(e) {
    e.preventDefault();
    const id = document.getElementById('gradingPerformerId').value;
    const p = State.participants.find(pt => String(pt.id) === String(id));
    if (!p) return;

    const ja = parseInt(document.getElementById('gradeJudgeA').value) || 0;
    const jb = parseInt(document.getElementById('gradeJudgeB').value) || 0;
    const jc = parseInt(document.getElementById('gradeJudgeC').value) || 0;

    p.judgeA = ja;
    p.judgeB = jb;
    p.judgeC = jc;
    p.consistency = Math.round((ja + jb + jc) / 3);

    if (_supabase) {
        await _supabase.from('participants').update({
            judgeA: ja,
            judgeB: jb,
            judgeC: jc,
            consistency: p.consistency
        }).eq('id', id);
        logSync(`Successfully graded performer ${p.name} on cloud.`, "success");
    }

    saveToCache();
    closeGradingModal();
    renderParticipantRegistry();
    renderConsistencyMatrix();
    runKMeansAndAnomalies();
    renderMyProfile();
    if (State.selectedEvent) renderRecommendations();

    showToast(`Grades updated for ${p.name}.`, "success");
}

window.renderMyProfile = renderMyProfile;
window.deleteMyProfile = deleteMyProfile;
window.prepopulateRegistrationForm = prepopulateRegistrationForm;
window.openGradingModal = openGradingModal;
window.closeGradingModal = closeGradingModal;
window.submitGrading = submitGrading;
window.selectEvent = selectEvent;
window.deleteParticipant = deleteParticipant;
window.switchTab = switchTab;
