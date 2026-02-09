#pragma once

#ifdef HELMSMAN_DEBUG_PROBE
#define PROBE_DUMP(data, path) arcforge::utils::FileUtils::GetInstance().dumpBinary(data, path)
#else
#define PROBE_DUMP(data, path)
#endif
