from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RAW_ROOT = Path("data/raw_samples/raw/abs")
OUTPUT_DIR = Path("data/parsed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "abs_parsed.parquet"
OUTPUT_CSV = OUTPUT_DIR / "abs_parsed.csv"


# ============================================================
# UTILITÁRIOS
# ============================================================

def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if s == "":
        return None
    if "," in s and "." not in s:
        s = s.replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return None


def to_int01(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0

    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "sim"}:
        return 1
    if s in {"0", "false", "no", "nao", "não"}:
        return 0
    return None


def parse_datetime(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value, format="ISO8601", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def classify_signal_quality(rssi_dbm: Optional[float]) -> Optional[str]:
    if rssi_dbm is None:
        return None
    if rssi_dbm >= -80:
        return "good"
    if rssi_dbm >= -90:
        return "moderate"
    if rssi_dbm >= -100:
        return "poor"
    return "critical"


def derive_machine_status(machine_on: Optional[int], machine_operating: Optional[int]) -> Optional[str]:
    if machine_on is None or machine_operating is None:
        return None
    if machine_on == 0:
        return "off"
    if machine_on == 1 and machine_operating == 0:
        return "idle"
    if machine_on == 1 and machine_operating == 1:
        return "operating"
    return None


def make_normalized_id() -> str:
    return str(uuid.uuid4())


def build_quality_flags(
    event_time: Optional[pd.Timestamp],
    device_id: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    machine_on: Optional[int],
    machine_operating: Optional[int],
    gps_valid: Optional[int],
) -> List[str]:
    flags: List[str] = []

    if event_time is None:
        flags.append("missing_event_time")
    if not device_id:
        flags.append("missing_device_id")

    if (latitude is None) ^ (longitude is None):
        flags.append("partial_position")

    if latitude is not None and not (-90 <= latitude <= 90):
        flags.append("invalid_latitude_range")

    if longitude is not None and not (-180 <= longitude <= 180):
        flags.append("invalid_longitude_range")

    if machine_operating == 1 and machine_on == 0:
        flags.append("operating_while_off")

    if gps_valid == 1 and (latitude is None or longitude is None):
        flags.append("gps_valid_without_position")

    return flags


def hex_to_bytes(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str)


def u8(b: bytes, offset: int) -> int:
    return int.from_bytes(b[offset:offset + 1], byteorder="big", signed=False)


def u16(b: bytes, offset: int) -> int:
    return int.from_bytes(b[offset:offset + 2], byteorder="big", signed=False)


def i16(b: bytes, offset: int) -> int:
    return int.from_bytes(b[offset:offset + 2], byteorder="big", signed=True)


def u32(b: bytes, offset: int) -> int:
    return int.from_bytes(b[offset:offset + 4], byteorder="big", signed=False)


# ============================================================
# LEITURA RAW
# ============================================================

def list_jsonl_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Pasta raw não encontrada: {root.resolve()}")
    return sorted(root.rglob("*.jsonl"))


def read_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_source_file"] = str(filepath)
                obj["_source_line"] = line_number
                rows.append(obj)
            except json.JSONDecodeError:
                rows.append({
                    "_source_file": str(filepath),
                    "_source_line": line_number,
                    "_line_parse_error": "invalid_jsonl_line",
                    "_raw_line": line,
                })

    return rows


# ============================================================
# DECODIFICAÇÃO DO FRAME ABS
# Frame esperado:
#  AA55 (2 bytes)
#  DEV_HASH (4)
#  EPOCH (4)
#  GPS_STATE (1)
#  LAT_INT (2, signed)
#  LAT_FRAC (2, signed)
#  LON_INT (2, signed)
#  LON_FRAC (2, signed)
#  SPEED_KNOTS (2)
#  RSSI_CODE (1)
#  IO_BITMAP (1)
#  EXT_V_X10 (2)
#  FLAGS (1)
#  CRC (2)
#
# Tamanho total esperado: 26 bytes = 52 hex chars
# ============================================================

EXPECTED_MIN_HEX_LEN = 52


def decode_abs_frame(raw_hex: str) -> Dict[str, Any]:
    result = {
        "frame_valid": 0,
        "frame_error": None,
        "dev_hash": None,
        "event_time": None,
        "gps_state": None,
        "latitude": None,
        "longitude": None,
        "speed_kmh": None,
        "rssi_dbm": None,
        "machine_on": None,
        "machine_operating": None,
        "is_buffered": None,
        "ext_v": None,
        "flags_raw": None,
        "crc_raw": None,
        "gps_valid": None,
        "gps_status": None,
        "flag_poor_coverage": 0,
        "flag_partial_obstruction": 0,
        "flag_metal_shed": 0,
        "flag_buffered": 0,
    }

    if raw_hex is None:
        result["frame_error"] = "missing_raw_hex"
        return result

    hex_str = str(raw_hex).strip().upper().replace(" ", "")
    if hex_str == "":
        result["frame_error"] = "empty_raw_hex"
        return result

    if len(hex_str) < EXPECTED_MIN_HEX_LEN:
        result["frame_error"] = "truncated_frame"
        return result

    try:
        b = hex_to_bytes(hex_str)
    except ValueError:
        result["frame_error"] = "invalid_hex"
        return result

    if len(b) < 26:
        result["frame_error"] = "frame_too_short"
        return result

    header = b[0:2].hex().upper()
    if header != "AA55":
        result["frame_error"] = "invalid_header"
        return result

    try:
        dev_hash = u32(b, 2)
        epoch = u32(b, 6)
        gps_state = u8(b, 10)

        lat_int = i16(b, 11)
        lat_frac = i16(b, 13)
        lon_int = i16(b, 15)
        lon_frac = i16(b, 17)

        speed_knots = u16(b, 19)
        rssi_code = u8(b, 21)
        io_bitmap = u8(b, 22)
        ext_v_x10 = u16(b, 23)
        flags = u8(b, 25)
        crc_raw = b[26:28].hex().upper() if len(b) >= 28 else None

        # timestamp
        event_time = pd.to_datetime(epoch, unit="s", utc=False, errors="coerce")

        # latitude/longitude
        latitude = lat_int + (lat_frac / 10000.0)
        longitude = lon_int + (lon_frac / 10000.0)

        # velocidade
        speed_kmh = float(speed_knots) * 1.85

        # rssi
        # no frame simulado, foi gravado abs(rssi_dbm), então reconstituímos como negativo
        rssi_dbm = -float(rssi_code)

        # io bitmap
        machine_on = 1 if (io_bitmap & 0b00000001) else 0
        machine_operating = 1 if (io_bitmap & 0b00000010) else 0
        is_buffered = 1 if (io_bitmap & 0b00000100) else 0

        # flags
        flag_poor_coverage = 1 if (flags & 0b00000001) else 0
        flag_partial_obstruction = 1 if (flags & 0b00000010) else 0
        flag_metal_shed = 1 if (flags & 0b00000100) else 0
        flag_buffered = 1 if (flags & 0b00001000) else 0

        # gps state
        if gps_state == 0:
            gps_valid = 1
            gps_status = "valid"
        elif gps_state == 1:
            gps_valid = 0
            gps_status = "lost"
        elif gps_state == 2:
            gps_valid = 0
            gps_status = "invalid"
        else:
            gps_valid = 0
            gps_status = "unknown"

        result.update({
            "frame_valid": 1,
            "frame_error": None,
            "dev_hash": dev_hash,
            "event_time": event_time,
            "gps_state": gps_state,
            "latitude": latitude,
            "longitude": longitude,
            "speed_kmh": speed_kmh,
            "rssi_dbm": rssi_dbm,
            "machine_on": machine_on,
            "machine_operating": machine_operating,
            "is_buffered": is_buffered,
            "ext_v": ext_v_x10 / 10.0,
            "flags_raw": flags,
            "crc_raw": crc_raw,
            "gps_valid": gps_valid,
            "gps_status": gps_status,
            "flag_poor_coverage": flag_poor_coverage,
            "flag_partial_obstruction": flag_partial_obstruction,
            "flag_metal_shed": flag_metal_shed,
            "flag_buffered": flag_buffered,
        })
        return result

    except Exception as e:
        result["frame_error"] = f"decode_exception:{type(e).__name__}"
        return result


# ============================================================
# PARSER ABS
# ============================================================

def parse_abs_record(envelope: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "normalized_id": make_normalized_id(),
        "hardware": "ABS",
        "device_id": None,
        "hardware_session_id": None,
        "session_id": None,
        "source_system": None,
        "payload_format": "hex",

        "state_client": None,
        "city": None,
        "execution_place": None,
        "seed_type": None,
        "weight_kg": None,

        "event_time": None,
        "ingestion_time": None,
        "event_date": None,
        "event_hour": None,
        "delay_ms": None,
        "is_late": None,
        "is_buffered": None,

        "latitude": None,
        "longitude": None,
        "latitude_ref": None,
        "longitude_ref": None,
        "gps_valid": None,
        "gps_status": None,
        "speed_kmh": None,
        "rssi_dbm": None,
        "signal_quality": None,
        "network_condition": None,

        "machine_on": None,
        "machine_operating": None,
        "machine_status": None,
        "scenario_state": None,

        "parse_ok": 0,
        "parse_error": None,
        "data_quality_ok": 0,
        "data_quality_flags": None,
        "raw_record_id": None,
        "raw_payload": None,
        "observacoes": None,

        # auxiliares ABS
        "dev_hash": None,
        "gps_state_raw": None,
        "ext_v": None,
        "flags_raw": None,
        "crc_raw": None,

        "_source_file": envelope.get("_source_file"),
        "_source_line": envelope.get("_source_line"),
    }

    if envelope.get("_line_parse_error"):
        normalized["parse_error"] = envelope["_line_parse_error"]
        normalized["raw_payload"] = envelope.get("_raw_line")
        normalized["data_quality_flags"] = "invalid_jsonl_line"
        return normalized

    # metadados do envelope
    normalized["source_system"] = envelope.get("source_system")
    normalized["hardware_session_id"] = envelope.get("hardware_session_id")
    normalized["raw_record_id"] = envelope.get("record_id")
    normalized["city"] = envelope.get("city")
    normalized["state_client"] = envelope.get("state_client")
    normalized["execution_place"] = envelope.get("execution_place")
    normalized["seed_type"] = envelope.get("seed_type")
    normalized["weight_kg"] = to_float(envelope.get("weight_kg"))
    normalized["latitude_ref"] = to_float(envelope.get("latitude_ref"))
    normalized["longitude_ref"] = to_float(envelope.get("longitude_ref"))
    normalized["delay_ms"] = to_float(envelope.get("delay_ms"))
    normalized["network_condition"] = envelope.get("network_condition")
    normalized["scenario_state"] = envelope.get("scenario_state")
    normalized["observacoes"] = envelope.get("observacoes")

    ingestion_time = parse_datetime(envelope.get("received_at"))
    normalized["ingestion_time"] = ingestion_time

    raw_payload = envelope.get("raw_payload")
    if raw_payload is None:
        normalized["parse_error"] = "missing_raw_payload"
        normalized["data_quality_flags"] = "missing_raw_payload"
        return normalized

    raw_hex = str(raw_payload).strip().upper()
    normalized["raw_payload"] = raw_hex

    decoded = decode_abs_frame(raw_hex)
    if decoded["frame_valid"] != 1:
        normalized["parse_error"] = decoded["frame_error"]
        normalized["data_quality_flags"] = decoded["frame_error"]
        return normalized

    device_id_from_envelope = envelope.get("device_id")
    if device_id_from_envelope:
        device_id = device_id_from_envelope
    else:
        device_id = f"ABS_{decoded['dev_hash']}"

    event_time = decoded["event_time"]
    latitude = decoded["latitude"]
    longitude = decoded["longitude"]
    gps_valid = decoded["gps_valid"]
    gps_status = decoded["gps_status"]
    speed_kmh = decoded["speed_kmh"]
    rssi_dbm = decoded["rssi_dbm"]
    machine_on = decoded["machine_on"]
    machine_operating = decoded["machine_operating"]
    is_buffered = decoded["is_buffered"]

    normalized["device_id"] = device_id
    normalized["event_time"] = event_time
    normalized["latitude"] = latitude
    normalized["longitude"] = longitude
    normalized["gps_valid"] = gps_valid
    normalized["gps_status"] = gps_status
    normalized["speed_kmh"] = speed_kmh
    normalized["rssi_dbm"] = rssi_dbm
    normalized["machine_on"] = machine_on
    normalized["machine_operating"] = machine_operating
    normalized["is_buffered"] = is_buffered

    normalized["dev_hash"] = decoded["dev_hash"]
    normalized["gps_state_raw"] = decoded["gps_state"]
    normalized["ext_v"] = decoded["ext_v"]
    normalized["flags_raw"] = decoded["flags_raw"]
    normalized["crc_raw"] = decoded["crc_raw"]

    if normalized["hardware_session_id"]:
        normalized["session_id"] = normalized["hardware_session_id"]
    elif device_id and event_time is not None:
        normalized["session_id"] = f"{device_id}_{event_time.strftime('%Y%m%d')}"
    elif device_id:
        normalized["session_id"] = f"{device_id}_unknown_session"

    if event_time is not None:
        normalized["event_date"] = event_time.date().isoformat()
        normalized["event_hour"] = int(event_time.hour)

    if normalized["delay_ms"] is not None:
        normalized["is_late"] = 1 if normalized["delay_ms"] > 5000 else 0

    normalized["signal_quality"] = classify_signal_quality(rssi_dbm)
    normalized["machine_status"] = derive_machine_status(machine_on, machine_operating)

    flags = build_quality_flags(
        event_time=event_time,
        device_id=device_id,
        latitude=latitude,
        longitude=longitude,
        machine_on=machine_on,
        machine_operating=machine_operating,
        gps_valid=gps_valid,
    )

    # flags específicas do ABS
    if decoded["flag_poor_coverage"] == 1:
        flags.append("poor_coverage")
    if decoded["flag_partial_obstruction"] == 1:
        flags.append("partial_obstruction")
    if decoded["flag_metal_shed"] == 1:
        flags.append("metal_shed")
    if decoded["flag_buffered"] == 1:
        flags.append("buffered_flag")

    normalized["parse_ok"] = 1
    normalized["parse_error"] = None
    normalized["data_quality_ok"] = 1 if len(flags) == 0 else 0
    normalized["data_quality_flags"] = ";".join(flags) if flags else None

    return normalized


# ============================================================
# EXECUÇÃO
# ============================================================

def main() -> None:
    files = list_jsonl_files(RAW_ROOT)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .jsonl encontrado em {RAW_ROOT.resolve()}")

    parsed_rows: List[Dict[str, Any]] = []

    print(f"Encontrados {len(files)} arquivo(s) JSONL em: {RAW_ROOT}")

    for file in files:
        print(f"Lendo: {file}")
        rows = read_jsonl(file)
        for row in rows:
            parsed_rows.append(parse_abs_record(row))

    df = pd.DataFrame(parsed_rows)

    if "event_time" in df.columns:
        df["event_time_dt"] = pd.to_datetime(df["event_time"], format="ISO8601", errors="coerce")
        df = df.sort_values(["event_time_dt", "device_id"], na_position="last").drop(columns=["event_time_dt"])

    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nSaídas geradas:")
    print(f"- {OUTPUT_PARQUET}")
    print(f"- {OUTPUT_CSV}")

    print("\nResumo:")
    print(f"Total de registros: {len(df)}")

    if "parse_ok" in df.columns:
        parse_ok_rate = df["parse_ok"].mean() * 100 if len(df) else 0
        print(f"Taxa de parse_ok: {parse_ok_rate:.2f}%")

    if "data_quality_ok" in df.columns:
        dq_ok_rate = df["data_quality_ok"].mean() * 100 if len(df) else 0
        print(f"Taxa de data_quality_ok: {dq_ok_rate:.2f}%")

    print("\nPrincipais parse_error:")
    if "parse_error" in df.columns:
        print(df["parse_error"].fillna("SEM_ERRO").value_counts().head(10).to_string())

    print("\nPrincipais data_quality_flags:")
    if "data_quality_flags" in df.columns:
        print(df["data_quality_flags"].fillna("SEM_FLAG").value_counts().head(15).to_string())


if __name__ == "__main__":
    main()