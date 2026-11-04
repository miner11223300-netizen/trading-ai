"""Rule-based multi-factor trading advisor.

Research tool only: scores confluence, does not promise profitable outcomes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .smc import analyze_smc


def _num(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _adx(df, period=14):
    high, low, close = df.high, df.low, df.close
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev = close.shift()
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean(), plus_di, minus_di


def _nearest_levels(df, lookback=120):
    d = df.tail(lookback)
    price = _num(d.close.iloc[-1])
    highs, lows = [], []
    for i in range(2, len(d)-2):
        h, l = _num(d.high.iloc[i]), _num(d.low.iloc[i])
        if h >= _num(d.high.iloc[i-2:i+3].max()): highs.append(h)
        if l <= _num(d.low.iloc[i-2:i+3].min()): lows.append(l)
    resistance = sorted(set(x for x in highs if x > price))
    support = sorted(set((x for x in lows if x < price)), reverse=True)
    return (support[0] if support else _num(d.low.min()),
            resistance[0] if resistance else _num(d.high.max()))


def analyze_market(df, capital=10000.0, risk_pct=0.5, rr=2.0, min_score=4):
    """Return human-readable trade decision, confluence and risk levels."""
    if df is None or len(df) < 60:
        raise ValueError("Для комплексного анализа требуется минимум 60 свечей.")
    d = df.copy().reset_index(drop=True)
    last = d.iloc[-1]
    price = _num(last.close)
    atr = _num(last.get('atr'))
    if not np.isfinite(atr) or atr <= 0:
        tr = pd.concat([(d.high-d.low), (d.high-d.close.shift()).abs(), (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
        atr = _num(tr.tail(14).mean(), price*0.01)
    adx_s, plus_di_s, minus_di_s = _adx(d)
    adx, plus_di, minus_di = _num(adx_s.iloc[-1], 0), _num(plus_di_s.iloc[-1], 0), _num(minus_di_s.iloc[-1], 0)
    rsi = _num(last.get('rsi'), 50)
    ema20, ema50, ema200 = (_num(last.get(k), price) for k in ('ema20','ema50','ema200'))
    macd, macd_sig = _num(last.get('macd'), 0), _num(last.get('macd_signal'), 0)
    vol = _num(last.get('volume'), 0)
    vol_avg = _num(last.get('vol_sma20'), 0)
    vol_ratio = vol / vol_avg if vol_avg > 0 else 1.0
    support, resistance = _nearest_levels(d)
    smc = analyze_smc(d)

    long_score = short_score = 0
    factors = []
    def add(name, long_pts, short_pts, detail):
        nonlocal long_score, short_score
        long_score += long_pts; short_score += short_pts
        factors.append({'factor': name, 'long': long_pts, 'short': short_pts, 'comment': detail})

    if price > ema20 > ema50:
        add('Краткосрочный тренд EMA', 2, 0, 'Цена выше EMA20, EMA20 выше EMA50')
    elif price < ema20 < ema50:
        add('Краткосрочный тренд EMA', 0, 2, 'Цена ниже EMA20, EMA20 ниже EMA50')
    else:
        add('Краткосрочный тренд EMA', 0, 0, 'EMA20/EMA50 не дают согласованного направления')
    if price > ema200: add('Долгосрочный фильтр EMA200', 1, 0, 'Цена выше EMA200')
    elif price < ema200: add('Долгосрочный фильтр EMA200', 0, 1, 'Цена ниже EMA200')
    else: add('Долгосрочный фильтр EMA200', 0, 0, 'Цена около EMA200')
    if macd > macd_sig: add('MACD', 1, 0, 'MACD выше сигнальной линии')
    elif macd < macd_sig: add('MACD', 0, 1, 'MACD ниже сигнальной линии')
    else: add('MACD', 0, 0, 'Нет перевеса MACD')
    if 52 <= rsi <= 68: add('RSI', 1, 0, f'RSI {rsi:.1f}: умеренный бычий импульс')
    elif 32 <= rsi <= 48: add('RSI', 0, 1, f'RSI {rsi:.1f}: умеренный медвежий импульс')
    elif rsi > 75: add('RSI', 0, 1, f'RSI {rsi:.1f}: перекупленность, риск отката')
    elif rsi < 25: add('RSI', 1, 0, f'RSI {rsi:.1f}: перепроданность, возможен отскок')
    else: add('RSI', 0, 0, f'RSI {rsi:.1f}: нейтральная зона')
    if adx >= 20 and plus_di > minus_di: add('ADX / направленность', 1, 0, f'ADX {adx:.1f}, +DI выше -DI')
    elif adx >= 20 and minus_di > plus_di: add('ADX / направленность', 0, 1, f'ADX {adx:.1f}, -DI выше +DI')
    else: add('ADX / направленность', 0, 0, f'ADX {adx:.1f}: тренд слабый или не подтверждён')
    if vol_ratio >= 1.3 and price >= _num(d.close.iloc[-2]): add('Объём', 1, 0, f'Объём {vol_ratio:.2f}x от среднего, свеча растущая')
    elif vol_ratio >= 1.3 and price < _num(d.close.iloc[-2]): add('Объём', 0, 1, f'Объём {vol_ratio:.2f}x от среднего, свеча падающая')
    else: add('Объём', 0, 0, f'Объём {vol_ratio:.2f}x от среднего; сильного подтверждения нет')
    structure = smc.get('structure', '')
    if 'bullish' in structure.lower(): add('Структура Smart Money', 2, 0, structure)
    elif 'bearish' in structure.lower(): add('Структура Smart Money', 0, 2, structure)
    else: add('Структура Smart Money', 0, 0, structure or 'Структура не определена')

    net = long_score - short_score
    if net >= min_score and long_score >= short_score + 2:
        side = 'LONG'
    elif net <= -min_score and short_score >= long_score + 2:
        side = 'SHORT'
    else:
        side = 'WAIT'

    # A heuristic strength score, deliberately not labelled as a probability.
    strength = min(95, int(50 + abs(net) * 6 + max(0, adx-20)*0.4))
    risk_cash = max(0.0, float(capital) * float(risk_pct) / 100)
    stop_dist = max(1.5 * atr, price * 0.002)
    entry = price
    if side == 'LONG':
        structural_stop = support - 0.15*atr if support < price else price-stop_dist
        stop = min(price-stop_dist, structural_stop)
        dist = price-stop
        target = price + dist*float(rr)
    elif side == 'SHORT':
        structural_stop = resistance + 0.15*atr if resistance > price else price+stop_dist
        stop = max(price+stop_dist, structural_stop)
        dist = stop-price
        target = price - dist*float(rr)
    else:
        stop = target = np.nan; dist = np.nan
    qty = risk_cash/dist if side != 'WAIT' and dist > 0 else 0.0
    # Guard against a target beyond a nearby opposing level being presented as clean.
    warnings = []
    if side == 'LONG' and resistance > price and target > resistance:
        warnings.append('Цель проходит выше ближайшего сопротивления: рассмотрите частичную фиксацию у сопротивления.')
    if side == 'SHORT' and support < price and target < support:
        warnings.append('Цель проходит ниже ближайшей поддержки: рассмотрите частичную фиксацию у поддержки.')
    if adx < 18: warnings.append('ADX ниже 18: рынок может находиться во флэте, сигналы тренда менее надёжны.')
    if vol_ratio < 0.7: warnings.append('Объём ниже 70% среднего: подтверждение движения слабое.')
    if rsi >= 75 or rsi <= 25: warnings.append('RSI в экстремальной зоне: не входите без подтверждения следующей свечой.')
    if side == 'WAIT':
        if abs(net) < min_score: wait_reason = f'Недостаточная согласованность факторов: перевес {net:+d} балла при пороге ±{min_score}.'
        else: wait_reason = 'Сигналы индикаторов и структуры противоречат друг другу.'
        triggers = []
        if price > ema20: triggers.append(f'Для LONG: закрытие выше локального сопротивления {resistance:.6g} и подтверждение объёмом.')
        else: triggers.append(f'Для LONG: возврат выше EMA20 ({ema20:.6g}) и подтверждение импульсом.')
        if price < ema20: triggers.append(f'Для SHORT: закрепление ниже локальной поддержки {support:.6g}.')
        else: triggers.append(f'Для SHORT: уход ниже EMA20 ({ema20:.6g}) с подтверждением объёмом.')
    else:
        wait_reason = ''
        triggers = [f'Вход рассматривается около закрытия последней свечи ({entry:.8g}); предпочтительно дождаться подтверждения следующей свечой.']
        triggers.append(f'Отмена сценария: закрытие за уровнем Stop Loss ({stop:.8g}) или изменение структуры в противоположную сторону.')

    return {
        'side': side, 'price': price, 'entry': entry, 'stop_loss': stop, 'take_profit': target,
        'quantity': qty, 'risk_cash': risk_cash, 'rr': float(rr), 'atr': atr,
        'long_score': long_score, 'short_score': short_score, 'net_score': net,
        'strength_index': strength, 'rsi': rsi, 'adx': adx, 'plus_di': plus_di,
        'minus_di': minus_di, 'volume_ratio': vol_ratio, 'support': support,
        'resistance': resistance, 'ema20': ema20, 'ema50': ema50, 'ema200': ema200,
        'structure': structure, 'fvg_count': len(smc.get('fvg', [])),
        'order_block_count': len(smc.get('order_blocks', [])), 'factors': factors,
        'warnings': warnings, 'wait_reason': wait_reason, 'triggers': triggers,
        'explanation': 'Итог сформирован по совокупности EMA, MACD, RSI, ADX/DI, объёма и структуры Smart Money. Индекс силы — эвристическая оценка согласованности, не вероятность прибыли.'
    }
