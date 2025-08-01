use alloc::{boxed::Box, format, string::String, sync::Arc, vec, vec::Vec};
use core::{
	ffi::c_void,
	fmt::{self, Debug},
	hash::Hash,
	marker::PhantomData,
	mem,
	ptr::{self, NonNull},
	slice
};
#[cfg(feature = "std")]
use std::collections::HashMap;

use super::{
	DowncastableTarget, DynValue, Value, ValueInner, ValueRef, ValueRefMut, ValueType, ValueTypeMarker,
	impl_tensor::{DynTensor, Tensor}
};
use crate::{
	AsPointer, ErrorCode,
	error::{Error, Result},
	memory::Allocator,
	ortsys,
	tensor::{IntoTensorElementType, PrimitiveTensorElementType, TensorElementType}
};

pub trait MapValueTypeMarker: ValueTypeMarker {
	private_trait!();
}

#[derive(Debug)]
pub struct DynMapValueType;
impl ValueTypeMarker for DynMapValueType {
	fn fmt(f: &mut fmt::Formatter) -> fmt::Result {
		f.write_str("DynMap")
	}

	private_impl!();
}
impl MapValueTypeMarker for DynMapValueType {
	private_impl!();
}

impl DowncastableTarget for DynMapValueType {
	fn can_downcast(dtype: &ValueType) -> bool {
		matches!(dtype, ValueType::Map { .. })
	}

	private_impl!();
}

#[derive(Debug)]
pub struct MapValueType<K: IntoTensorElementType + Clone + Hash + Eq, V: IntoTensorElementType + Debug>(PhantomData<(K, V)>);
impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: IntoTensorElementType + Debug> ValueTypeMarker for MapValueType<K, V> {
	fn fmt(f: &mut fmt::Formatter) -> fmt::Result {
		f.write_str("Map<")?;
		<TensorElementType as fmt::Display>::fmt(&K::into_tensor_element_type(), f)?;
		f.write_str(", ")?;
		<TensorElementType as fmt::Display>::fmt(&V::into_tensor_element_type(), f)?;
		f.write_str(">")
	}

	private_impl!();
}
impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: IntoTensorElementType + Debug> MapValueTypeMarker for MapValueType<K, V> {
	private_impl!();
}

impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: IntoTensorElementType + Debug> DowncastableTarget for MapValueType<K, V> {
	fn can_downcast(dtype: &ValueType) -> bool {
		match dtype {
			ValueType::Map { key, value } => *key == K::into_tensor_element_type() && *value == V::into_tensor_element_type(),
			_ => false
		}
	}

	private_impl!();
}

pub type DynMap = Value<DynMapValueType>;
pub type Map<K, V> = Value<MapValueType<K, V>>;

pub type DynMapRef<'v> = ValueRef<'v, DynMapValueType>;
pub type DynMapRefMut<'v> = ValueRefMut<'v, DynMapValueType>;
pub type MapRef<'v, K, V> = ValueRef<'v, MapValueType<K, V>>;
pub type MapRefMut<'v, K, V> = ValueRefMut<'v, MapValueType<K, V>>;

impl<Type: MapValueTypeMarker + ?Sized> Value<Type> {
	pub fn try_extract_key_values<K: IntoTensorElementType + Clone + Hash + Eq, V: PrimitiveTensorElementType + Clone>(&self) -> Result<Vec<(K, V)>> {
		match self.dtype() {
			ValueType::Map { key, value } => {
				let k_type = K::into_tensor_element_type();
				if k_type != *key {
					return Err(Error::new_with_code(ErrorCode::InvalidArgument, format!("Cannot extract Map<{:?}, _> (value has K type {:?})", k_type, key)));
				}
				let v_type = V::into_tensor_element_type();
				if v_type != *value {
					return Err(Error::new_with_code(
						ErrorCode::InvalidArgument,
						format!("Cannot extract Map<{}, {}> from Map<{}, {}>", K::into_tensor_element_type(), V::into_tensor_element_type(), k_type, v_type)
					));
				}

				let allocator = Allocator::default();

				let mut key_tensor_ptr = ptr::null_mut();
				ortsys![unsafe GetValue(self.ptr(), 0, allocator.ptr().cast_mut(), &mut key_tensor_ptr)?; nonNull(key_tensor_ptr)];
				let key_value: DynTensor = unsafe { Value::from_ptr(key_tensor_ptr, None) };
				if K::into_tensor_element_type() != TensorElementType::String {
					let dtype = key_value.dtype();
					let (key_tensor_shape, key_tensor) = match dtype {
						ValueType::Tensor { ty, shape, .. } => {
							let mem = key_value.memory_info();
							if !mem.is_cpu_accessible() {
								return Err(Error::new(format!(
									"Cannot extract from value on device `{}`, which is not CPU accessible",
									mem.allocation_device().as_str()
								)));
							}

							if *ty == K::into_tensor_element_type() {
								let mut output_array_ptr: *mut K = ptr::null_mut();
								let output_array_ptr_ptr: *mut *mut K = &mut output_array_ptr;
								let output_array_ptr_ptr_void: *mut *mut c_void = output_array_ptr_ptr.cast();
								ortsys![unsafe GetTensorMutableData(key_tensor_ptr.as_ptr(), output_array_ptr_ptr_void)?];
								if output_array_ptr.is_null() {
									output_array_ptr = NonNull::dangling().as_ptr();
								}

								(shape, unsafe { slice::from_raw_parts(output_array_ptr, shape.num_elements()) })
							} else {
								return Err(Error::new_with_code(
									ErrorCode::InvalidArgument,
									format!(
										"Cannot extract Map<{}, {}> from Map<{}, {}>",
										K::into_tensor_element_type(),
										V::into_tensor_element_type(),
										k_type,
										v_type
									)
								));
							}
						}
						_ => unreachable!()
					};

					let mut value_tensor_ptr = ptr::null_mut();
					ortsys![unsafe GetValue(self.ptr(), 1, allocator.ptr().cast_mut(), &mut value_tensor_ptr)?; nonNull(value_tensor_ptr)];
					let value_value: DynTensor = unsafe { Value::from_ptr(value_tensor_ptr, None) };
					let (value_tensor_shape, value_tensor) = value_value.try_extract_tensor::<V>()?;

					assert_eq!(key_tensor_shape.len(), 1);
					assert_eq!(value_tensor_shape.len(), 1);
					assert_eq!(key_tensor_shape[0], value_tensor_shape[0]);

					let mut vec = Vec::with_capacity(key_tensor_shape[0] as _);
					for i in 0..key_tensor_shape[0] as usize {
						vec.push((key_tensor[i].clone(), value_tensor[i].clone()));
					}
					Ok(vec)
				} else {
					let (key_tensor_shape, key_tensor) = key_value.try_extract_strings()?;
					// SAFETY: `IntoTensorElementType` is a private trait, and we only map the `String` type to `TensorElementType::String`,
					// so at this point, `K` is **always** the `String` type, and this transmute really does nothing but please the type
					// checker.
					let key_tensor: Vec<K> = unsafe { mem::transmute(key_tensor) };

					let mut value_tensor_ptr = ptr::null_mut();
					ortsys![unsafe GetValue(self.ptr(), 1, allocator.ptr().cast_mut(), &mut value_tensor_ptr)?; nonNull(value_tensor_ptr)];
					let value_value: DynTensor = unsafe { Value::from_ptr(value_tensor_ptr, None) };
					let (value_tensor_shape, value_tensor) = value_value.try_extract_tensor::<V>()?;

					assert_eq!(key_tensor_shape.len(), 1);
					assert_eq!(value_tensor_shape.len(), 1);
					assert_eq!(key_tensor_shape[0], value_tensor_shape[0]);

					let mut vec = Vec::with_capacity(key_tensor_shape[0] as _);
					for i in 0..key_tensor_shape[0] as usize {
						vec.push((key_tensor[i].clone(), value_tensor[i].clone()));
					}
					Ok(vec.into_iter().collect())
				}
			}
			t => Err(Error::new_with_code(
				ErrorCode::InvalidArgument,
				format!("Cannot extract Map<{}, {}> from {t}", K::into_tensor_element_type(), V::into_tensor_element_type())
			))
		}
	}

	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn try_extract_map<K: IntoTensorElementType + Clone + Hash + Eq, V: PrimitiveTensorElementType + Clone>(&self) -> Result<HashMap<K, V>> {
		self.try_extract_key_values().map(|c| c.into_iter().collect())
	}
}

impl<K: PrimitiveTensorElementType + Debug + Clone + Hash + Eq + 'static, V: PrimitiveTensorElementType + Debug + Clone + 'static> Value<MapValueType<K, V>> {
	/// Creates a [`Map`] from an iterable emitting `K` and `V`.
	///
	/// ```
	/// # use std::collections::HashMap;
	/// # use ort::value::Map;
	/// # fn main() -> ort::Result<()> {
	/// let mut map = HashMap::<i64, f32>::new();
	/// map.insert(0, 1.0);
	/// map.insert(1, 2.0);
	/// map.insert(2, 3.0);
	///
	/// let value = Map::<i64, f32>::new(map)?;
	///
	/// assert_eq!(*value.extract_map().get(&0).unwrap(), 1.0);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn new(data: impl IntoIterator<Item = (K, V)>) -> Result<Self> {
		let (keys, values): (Vec<K>, Vec<V>) = data.into_iter().unzip();
		Self::new_kv(Tensor::from_array((vec![keys.len()], keys))?, Tensor::from_array((vec![values.len()], values))?)
	}
}

impl<V: PrimitiveTensorElementType + Debug + Clone + 'static> Value<MapValueType<String, V>> {
	/// Creates a [`Map`] from an iterable emitting `K` and `V`.
	///
	/// ```
	/// # use std::collections::HashMap;
	/// # use ort::value::Map;
	/// # fn main() -> ort::Result<()> {
	/// let mut map = HashMap::<String, f32>::new();
	/// map.insert("one".to_string(), 1.0);
	/// map.insert("two".to_string(), 2.0);
	/// map.insert("three".to_string(), 3.0);
	///
	/// let value = Map::<String, f32>::new(map)?;
	///
	/// assert_eq!(*value.extract_map().get("one").unwrap(), 1.0);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn new(data: impl IntoIterator<Item = (String, V)>) -> Result<Self> {
		let (keys, values): (Vec<String>, Vec<V>) = data.into_iter().unzip();
		Self::new_kv(Tensor::from_string_array((vec![keys.len()], keys.as_slice()))?, Tensor::from_array((vec![values.len()], values))?)
	}
}

impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq + 'static, V: IntoTensorElementType + Debug + Clone + 'static> Value<MapValueType<K, V>> {
	/// Creates a [`Map`] from two tensors of keys & values respectively.
	///
	/// ```
	/// # use std::collections::HashMap;
	/// # use ort::value::{Map, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let keys = Tensor::<i64>::from_array(([4], vec![0, 1, 2, 3]))?;
	/// let values = Tensor::<f32>::from_array(([4], vec![1., 2., 3., 4.]))?;
	///
	/// let value = Map::new_kv(keys, values)?;
	///
	/// assert_eq!(*value.extract_map().get(&0).unwrap(), 1.0);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn new_kv(keys: Tensor<K>, values: Tensor<V>) -> Result<Self> {
		let mut value_ptr = ptr::null_mut();
		let values: [DynValue; 2] = [keys.into_dyn(), values.into_dyn()];
		let value_ptrs: Vec<*const ort_sys::OrtValue> = values.iter().map(|c| c.ptr()).collect();
		ortsys![
			unsafe CreateValue(value_ptrs.as_ptr(), 2, ort_sys::ONNXType::ONNX_TYPE_MAP, &mut value_ptr)?;
			nonNull(value_ptr)
		];
		Ok(Value {
			inner: Arc::new(ValueInner {
				ptr: value_ptr,
				dtype: ValueType::Map {
					key: K::into_tensor_element_type(),
					value: V::into_tensor_element_type()
				},
				drop: true,
				memory_info: None,
				_backing: Some(Box::new(values))
			}),
			_markers: PhantomData
		})
	}
}

impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: PrimitiveTensorElementType + Debug + Clone> Value<MapValueType<K, V>> {
	pub fn extract_key_values(&self) -> Vec<(K, V)> {
		self.try_extract_key_values().expect("Failed to extract map")
	}

	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn extract_map(&self) -> HashMap<K, V> {
		self.try_extract_map().expect("Failed to extract map")
	}
}

impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: IntoTensorElementType + Debug + Clone> Value<MapValueType<K, V>> {
	/// Converts from a strongly-typed [`Map<K, V>`] to a type-erased [`DynMap`].
	#[inline]
	pub fn upcast(self) -> DynMap {
		unsafe { self.transmute_type() }
	}

	/// Converts from a strongly-typed [`Map<K, V>`] to a reference to a type-erased [`DynMap`].
	#[inline]
	pub fn upcast_ref(&self) -> DynMapRef<'_> {
		DynMapRef::new(Value {
			inner: Arc::clone(&self.inner),
			_markers: PhantomData
		})
	}

	/// Converts from a strongly-typed [`Map<K, V>`] to a mutable reference to a type-erased [`DynMap`].
	#[inline]
	pub fn upcast_mut(&mut self) -> DynMapRefMut<'_> {
		DynMapRefMut::new(Value {
			inner: Arc::clone(&self.inner),
			_markers: PhantomData
		})
	}
}
