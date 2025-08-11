
#include <trajopt/typedefs.hpp>

namespace trajopt
{
#ifdef USE_THREAD_LOCAL
thread_local tesseract_common::TransformMap TrajOptVectorOfVector::transforms_cache;  // NOLINT
thread_local tesseract_common::TransformMap TrajOptMatrixOfVector::transforms_cache;  // NOLINT
#else
boost::thread_specific_ptr<tesseract_common::TransformMap> TrajOptVectorOfVector::transforms_cache_ptr;  // NOLINT
boost::thread_specific_ptr<tesseract_common::TransformMap> TrajOptMatrixOfVector::transforms_cache_ptr;  // NOLINT
#endif
}  // namespace trajopt
