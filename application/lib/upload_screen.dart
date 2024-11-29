import 'dart:convert';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class UploadScreen extends StatefulWidget {
  @override
  _UploadScreenState createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  String? _selectedFilePath;
  String _predictionResult = "";

  Future<void> _pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['xlsx'],
    );

    if (result != null) {
      setState(() {
        _selectedFilePath = result.files.single.path;
      });
    }
  }

  Future<void> _uploadAndPredict() async {
    if (_selectedFilePath == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Por favor, selecciona un archivo primero.")),
      );
      return;
    }

    var uri =
        Uri.parse('http://127.0.0.1:5001/predict'); // Cambia a la URL de tu API
    var request = http.MultipartRequest('POST', uri);

    request.files.add(
      await http.MultipartFile.fromPath('file', _selectedFilePath!),
    );

    var response = await request.send();

    if (response.statusCode == 200) {
      var responseBody = await http.Response.fromStream(response);
      var decodedResponse = json.decode(responseBody.body);
      setState(() {
        _predictionResult = decodedResponse['message'];
      });
    } else {
      setState(() {
        _predictionResult = "Error al realizar la predicción.";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Subir Archivo y Predecir"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ElevatedButton(
              onPressed: _pickFile,
              child: Text("Seleccionar Archivo Excel"),
            ),
            if (_selectedFilePath != null)
              Text(
                "Archivo seleccionado: ${_selectedFilePath!.split('/').last}",
                style: TextStyle(fontSize: 16),
              ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _uploadAndPredict,
              child: Text("Subir y Predecir"),
            ),
            SizedBox(height: 20),
            Text(
              "Resultado de la predicción:",
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            Text(
              _predictionResult,
              style: TextStyle(fontSize: 16, color: Colors.blue),
            ),
          ],
        ),
      ),
    );
  }
}
